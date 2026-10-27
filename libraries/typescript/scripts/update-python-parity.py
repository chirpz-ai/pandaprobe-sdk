"""Execute the unchanged Python source to generate cross-language regression cases.
Run at repo root: libraries/python/.venv/bin/python -B libraries/typescript/scripts/update-python-parity.py
"""
import asyncio
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root / 'libraries/python/src'), str(root / 'libraries/python/examples/reflect')]
from _support import scenario, refund_incident as refund, delivery, runner, failure_runner
from pandaprobe.api.models import TracePage, Score, Trace
from pandaprobe.reflect import ReflectConfig
from pandaprobe.reflect.shell import ReflectShell
from pandaprobe.reflect import models
os.environ.pop('REFLECT_EXAMPLE_MODEL', None)

def accepted(fn):
    try: return {'accepted': True, 'value': fn()}
    except (TypeError, ValueError): return {'accepted': False}

def support(ticket):
    desk = scenario.SupportDesk(ticket)
    stages = []
    for stage in ['inspect', 'resolve', 'repeat']:
        desk.begin(stage)
        output = [desk.lookup_order(ticket.id), desk.search_policy('current returns')]
        if stage != 'inspect': output.append(desk.resolve_ticket(**ticket.expected, idempotency_key=ticket.id))
        stages.append({'stage': stage, 'prompt': ticket.prompt(stage), 'outputs': output,
                       'verification': asdict(desk.verify('Recorded result.'))})
    return {'ticket': asdict(ticket), 'stages': stages, 'events': desk.events, 'ledger': desk.ledger}

def refund_run(order, amount):
    batch = refund.RefundBatch(order)
    outputs = [batch.lookup_refund_order(order.id), batch.submit_refund(order.id, amount, order.id),
               batch.submit_refund(order.id, amount, order.id)]
    for event in batch.events: event.pop('attempted_at', None)
    return {'order': asdict(order), 'amount': amount, 'outputs': outputs, 'events': batch.events,
            'verification': asdict(batch.verify('Recorded result.'))}

rule = '# Refund\nApplies to: return refunds\nExclude shipping.\nUse merchandise paid.\n'
files = {'refund.md': rule}
numbered = '\n'.join(f'/rules/refund.md:{i}:{line}' for i, line in enumerate(rule.splitlines(), 1))
audit_texts = [rule, numbered, '/rules/refund.md:3:Exclude shipping.', 'Exclude shipping.',
              numbered.replace('shipping.', 'shipping!'), numbered.replace('/rules/', 'rules/'),
              numbered.replace('/rules/', '/other/rules/'), 'Observation: '+numbered,
              '/rules/refund.md:999999999999999999:x', '# Refund', 'No rules',
              numbered+'\n'+numbered, '/rules/refund.md:3:Exclude shipping.\n--\n/rules/refund.md:4:Use merchandise paid.']

def config_in(file):
    tree = ast.parse(Path(file).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'ReflectConfig']
    values = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if k.arg != 'workspace'}
    return values

fixture = {
 'support_instructions': scenario.INSTRUCTIONS, 'refund_instructions': refund.INSTRUCTIONS,
 'support': [support(t) for t in scenario.tickets()],
 'round_two': [asdict(t) for t in scenario.tickets(1)],
 'support_cases': {p: [asdict(c) for c in scenario.replay_cases(p)] for p in ['returns-v1','returns-v2']},
 'refund_cases': [asdict(c) for c in refund.replay_cases()],
 'refund_prompts': [o.prompt for o in [refund.DISCOVERY, refund.FOLLOW_UP]],
 'refund': [refund_run(refund.DISCOVERY, n) for n in [7200,6500,True,6500.5]],
 'arguments': [],
 'example_config': config_in(root/'libraries/python/examples/reflect/_support/runner.py'),
 'failure_config': config_in(root/'libraries/python/examples/reflect/_support/failure_runner.py'),
 'config_defaults': {k: v for k, v in asdict(ReflectConfig()).items() if k != 'workspace'},
 'audit': [{'text': text, 'files': files, 'coverage': {p:asdict(r) for p,r in delivery.match_rules(text,files).items()}} for text in audit_texts],
 'api_counts': [], 'numeric_scores': [], 'api_nulls': [],
}
for failure in [False, True]:
    for argv in [[], ['--model','fixture','--task-timeout','1.25'], *([] if failure else [['--rounds','3','--max-tickets','2','--max-validations','0']])]:
        args = failure_runner.arguments(argv,framework='langchain',default_model='gpt-6-luna') if failure else runner.arguments('langchain','gpt-6-luna',argv)
        fixture['arguments'].append({'failure':failure,'argv':argv,'value':vars(args)})
for value in [0,1,True,False,1.0,1.2,'1',' 1 ','1.0','1.00','1e0','1_0','+1','01','١','',None,'-1','0x10']:
    fixture['api_counts'].append({'input':value,**accepted(lambda:TracePage(items=[],total=value,limit=1,offset=0).total)})
for value in ['1_0','1__0','_1','1_',' 1.25 ','١.٢٥','１.２５','1e2','1_0.2_5e-1','0x10','nan','inf','\u00851\u0085','\ufeff1','\x1c1',None,'','-0.0','0.5']:
    score=Score(id='00000000-0000-0000-0000-000000000001',trace_id='00000000-0000-0000-0000-000000000002',name='fixture',value=value,status='SUCCESS',data_type='NUMERIC')
    fixture['numeric_scores'].append({'input':value,'value':score.numeric_value})
base={'trace_id':'00000000-0000-0000-0000-000000000001','project_id':'00000000-0000-0000-0000-000000000002','name':'task','status':'COMPLETED','started_at':'2026-01-01T00:00:00Z'}
for field in ['tags','metadata','spans','session_id','ended_at','input','output']:
    row={**base,field:None}
    fixture['api_nulls'].append({'input':row,**accepted(lambda:Trace.model_validate(row).model_dump(mode='json'))})
fixture['records'] = []
record_inputs = {
    'RuleProposal': {'id':'proposal','created_at':'fixed','base_revision':'base','changes':{},'fingerprint':'fp','session_id':'s','execution_id':'e'},
    'ProposalOutcome': {'status':'no_proposal'},
    'ValidationVerdict': {'proposal_id':'proposal','outcome':'pending','reason':'reason','mode':'replay','created_at':'fixed'},
    'FailureNotice': {'id':'notice','created_at':'fixed','execution_id':'e','session_id':'s','trace_ids':[],'scores':{},'signatures':[]},
    'EvaluationOutcome': {'execution_id':'e','status':'evaluated'},
    'ExecutionReport': {'execution_id':'e','session_id':'s','task_status':'completed','trace_ids':[]},
}
for name, base in record_inputs.items():
    model = getattr(models, name)
    inputs = [base, {**base, 'unexpected':True}]
    inputs += [{**base, key: None} for key in model.model_fields]
    inputs += [{**base, key: 10} for key in model.model_fields]
    for row in inputs:
        fixture['records'].append({'model':name,'input':row,**accepted(lambda:model.model_validate(row).model_dump(mode='json'))})
fixture['rules'] = []
for text in [rule, '\u0085'+rule+'\x1c', '\ufeff'+rule, rule.replace('\n','\r'), rule.replace('\n','\u2028'), rule.replace('return refunds','a'), rule.replace('Applies to: ','Applies to:\u0085')]:
    row={'path':'refund.md','text':text}
    fixture['rules'].append({'input':row,**accepted(lambda:models.Rule.model_validate(row).model_dump())})
async def shell():
    result=[]
    for command in ['echo manual >/context.md; exit 124','echo interrupted >/context.md; sleep 5']:
        s=ReflectShell(ReflectConfig(shell_timeout=.01,shell_max_loop_iterations=10000000),{})
        try: result.append({'command':command,'output':await s.execute(command),'context':s.context()})
        finally: await s.aclose()
    return result
fixture['shell_results']=asyncio.run(shell())
(root/'libraries/typescript/test/fixtures/python-behavior.json').write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n')
print('Generated Python business, example, audit, API and native-shell parity fixtures')
