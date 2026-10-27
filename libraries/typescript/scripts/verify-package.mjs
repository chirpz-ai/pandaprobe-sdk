/** Verify actual npm artifacts in isolated consumers; never publish. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "pandaprobe-package-"));
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
function run(command, args, cwd = temporary) {
  return execFileSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    shell: process.platform === "win32" && command.endsWith(".cmd"),
  });
}
try {
  const [packed] = JSON.parse(run(npm, ["pack", "--json", "--pack-destination", temporary], root));
  assert(
    packed.files.every((f) => /^(dist\/|README.md$|LICENSE$|package.json$)/.test(f.path)),
    "unexpected runtime artifact",
  );
  assert(
    !packed.files.some((f) => /(node_modules|\.pandaprobe|\.env|\.py$|test\/|examples\/)/.test(f.path)),
    "private/development files packed",
  );
  writeFileSync(join(temporary, "package.json"), JSON.stringify({ private: true, type: "module" }));
  run(npm, [
    "install",
    "--ignore-scripts",
    "--no-audit",
    "--no-fund",
    "--omit=optional",
    join(temporary, packed.filename),
  ]);
  const installed = JSON.parse(readFileSync(join(temporary, "node_modules/pandaprobe/package.json"), "utf8"));
  assert.equal(Object.keys(installed.dependencies ?? {}).length, 0);
  const minimal = `import assert from 'node:assert/strict';import {init} from 'pandaprobe';import {Reflect as ReflectRuntime} from 'pandaprobe/reflect';import {PlatformClient} from 'pandaprobe/api';const client=init({apiKey:'fixture',projectName:'fixture',endpoint:'http://unused',enabled:false});const api=new PlatformClient(client);await api.close();assert.throws(()=>new ReflectRuntime(client),/Install Reflect runtimes/);await client.shutdown();console.log('minimal imports and missing-runtime error passed');`;
  writeFileSync(join(temporary, "minimal.mjs"), minimal);
  process.stdout.write(run(process.execPath, ["minimal.mjs"]));
  // Reflect's optional peers must not force upgrades for existing tracing-only users.
  const legacy = join(temporary, "legacy-tracing");
  mkdirSync(legacy);
  writeFileSync(join(legacy, "package.json"), JSON.stringify({ private: true, type: "module" }));
  run(
    npm,
    ["install", "--ignore-scripts", "--no-audit", "--no-fund", join(temporary, packed.filename), "ai@5.0.100"],
    legacy,
  );
  writeFileSync(
    join(legacy, "consumer.mjs"),
    `import assert from 'node:assert/strict';import {init} from 'pandaprobe';import {generateText,wrapLanguageModel} from 'ai';import {pandaProbeMiddleware} from 'pandaprobe/integrations/vercel-ai';
    const exports=[];globalThis.fetch=async(url,options)=>{exports.push(JSON.parse(options.body));return new Response('{}',{status:200});};const client=init({apiKey:'fixture',projectName:'fixture',endpoint:'http://fixture',flushInterval:100});
    try{const model=wrapLanguageModel({model:{specificationVersion:'v2',provider:'fixture',modelId:'fixture',supportedUrls:{},doGenerate:async()=>({content:[{type:'text',text:'legacy tracing'}],finishReason:'stop',usage:{inputTokens:1,outputTokens:2,totalTokens:3},warnings:[]})},middleware:pandaProbeMiddleware()});const result=await client.trace('legacy').run(()=>generateText({model,prompt:'hello'}));assert.equal(result.text,'legacy tracing');await client.flush();assert.equal(exports.length,1);assert.equal(exports[0].spans.length,1);assert.equal(exports[0].spans[0].output.messages[0].content,'legacy tracing');console.log('AI SDK 5 install and native tracing middleware passed');}finally{await client.shutdown();}`,
  );
  process.stdout.write(run(process.execPath, ["consumer.mjs"], legacy));
  // Runtime optional dependencies are actual installed artifacts, not links into the source checkout.
  run(npm, [
    "install",
    "--ignore-scripts",
    "--no-audit",
    "--no-fund",
    "@everruns/bashkit@0.19.0",
    "fs-native-extensions@1.5.1",
  ]);
  const body = `const assert=require('node:assert/strict');const sdk=require('pandaprobe');const {Reflect:ReflectRuntime}=require('pandaprobe/reflect');const {PlatformClient}=require('pandaprobe/api');const {wrapOpenAI}=require('pandaprobe/wrappers/openai');
  async function main(){const exports=[];globalThis.fetch=async(url,options)=>{exports.push(JSON.parse(options.body));return new Response('{}',{status:200});};const client=sdk.init({apiKey:'fixture',projectName:'fixture',endpoint:'http://fixture',flushInterval:100});const reflect=new ReflectRuntime(undefined,{config:{workspace:'workspace-FORMAT',evaluationSampleRate:0}});assert.equal(reflect.client,client,'subpaths must share initialized client');const api=new PlatformClient(client);await api.close();
  const provider={chat:{completions:{create:async()=>({id:'fixture',model:'fixture',choices:[{message:{role:'assistant',content:'done'}}],usage:{prompt_tokens:1,completion_tokens:1,total_tokens:2}})}}};wrapOpenAI(provider);let id;await reflect.execution({sessionId:'packed',executionId:'packed'},async e=>{id=e.id;const output=await reflect.bash("printf 'persistent memory' > /context.md; cat /context.md");assert.match(output,/persistent memory/);await client.trace('packed').run(async()=>{await provider.chat.completions.create({model:'fixture',messages:[{role:'user',content:'hello'}]});});});await reflect.waitExport();const report=reflect.report(id);assert.equal(report.trace_ids.length,1);assert.equal(exports[0].spans.length,1,'wrapper and root must share trace context');assert.equal(exports[0].metadata['pandaprobe.reflect'].role,'task');assert.equal((await reflect.settleEvaluation(id)).status,'skipped');await reflect.close();await client.shutdown();console.log('FORMAT native runtime, capture and wrapper context passed');}main().catch(e=>{console.error(e);process.exitCode=1;});`;
  writeFileSync(join(temporary, "consumer.cjs"), body.replaceAll("FORMAT", "cjs"));
  const esm = body.replace(
    "const assert=require('node:assert/strict');const sdk=require('pandaprobe');const {Reflect:ReflectRuntime}=require('pandaprobe/reflect');const {PlatformClient}=require('pandaprobe/api');const {wrapOpenAI}=require('pandaprobe/wrappers/openai');",
    "import assert from 'node:assert/strict';import * as sdk from 'pandaprobe';import {Reflect as ReflectRuntime} from 'pandaprobe/reflect';import {PlatformClient} from 'pandaprobe/api';import {wrapOpenAI} from 'pandaprobe/wrappers/openai';",
  );
  writeFileSync(join(temporary, "consumer.mjs"), esm.replaceAll("FORMAT", "esm"));
  for (const file of ["consumer.mjs", "consumer.cjs"]) process.stdout.write(run(process.execPath, [file]));
  const types = `import {Client} from 'pandaprobe';import {Reflect as ReflectRuntime,Outcome,ReplayCase,Invocation} from 'pandaprobe/reflect';import {PlatformClient} from 'pandaprobe/api';const client=new Client();const runtime=new ReflectRuntime(client);const api=new PlatformClient(client);const cases=[new ReplayCase({id:'case',input:{value:1},protected:true})];async function agent(request:Invocation){return request.instructions;}runtime.run(agent,{}, {sessionId:'typecheck'});new Outcome({scores:{task_completion:1},success:true});void cases;void api;`;
  const vercelTypes = `import {wrapVercelAgent} from 'pandaprobe/integrations/vercel-ai';
  class NativeAgent { async generate(input:{prompt:string}) { return {text:input.prompt}; } }
  const WrappedAgent=wrapVercelAgent(NativeAgent,{client});
  const wrappedAgent=wrapVercelAgent(new NativeAgent(),{client});
  const generated:Promise<{text:string}>=new WrappedAgent().generate({prompt:'typed'});
  const generatedInstance:Promise<{text:string}>=wrappedAgent.generate({prompt:'typed'});
  void generated;void generatedInstance;`;
  for (const extension of ["mts", "cts"])
    writeFileSync(join(temporary, `types.${extension}`), `${types}\n${vercelTypes}`);
  mkdirSync(join(temporary, "node_modules/@types"), { recursive: true });
  symlinkSync(
    resolve(root, "node_modules/@types/node"),
    join(temporary, "node_modules/@types/node"),
    process.platform === "win32" ? "junction" : "dir",
  );
  run(process.execPath, [
    resolve(root, "node_modules/typescript/bin/tsc"),
    "--noEmit",
    "--strict",
    "--target",
    "ES2022",
    "--module",
    "NodeNext",
    "--moduleResolution",
    "NodeNext",
    "types.mts",
    "types.cts",
  ]);
  console.log(
    `Packed package verified: ${packed.filename}; ${packed.files.length} files; ${packed.size} bytes; ESM/CJS/NodeNext declarations`,
  );
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
