/** Platform failures remain distinct from an ambiguous, potentially billable create. */
export class PlatformError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}
export class ApiError extends PlatformError {
  constructor(
    message: string,
    readonly statusCode: number,
    readonly requestId: string | null = null,
  ) {
    super(message);
  }
}
export class AuthenticationError extends ApiError {}
export class NotFoundError extends ApiError {}
export class RateLimitError extends ApiError {}
export class TransportError extends PlatformError {}
export class ApiDeadlineExceeded extends PlatformError {}
export class InvalidResponseError extends PlatformError {}
export class CreationOutcomeUnknown extends PlatformError {}
export class EvaluationFailed extends PlatformError {
  constructor(
    readonly runId: string,
    readonly status: string,
  ) {
    super(`Evaluation ${runId} ended with status ${status}`);
  }
}
export class InsufficientEvidence extends PlatformError {
  constructor(
    readonly runId: string,
    readonly missing: [string, string][],
  ) {
    super(`Evaluation ${runId} has ${missing.length} missing or unusable scores`);
  }
}
