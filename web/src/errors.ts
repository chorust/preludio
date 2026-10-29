export type PreludioErrorCategory =
  | "ModelUnavailable"
  | "IncompatibleInput"
  | "ExecutionFailed"
  | "InvalidOutput";

const categories: readonly PreludioErrorCategory[] = [
  "ModelUnavailable",
  "IncompatibleInput",
  "ExecutionFailed",
  "InvalidOutput",
];

export class PreludioError extends Error {
  readonly category: PreludioErrorCategory;
  readonly detail: unknown;

  constructor(category: PreludioErrorCategory, detail: unknown, message?: string) {
    super(message ?? `Preludio ${category}`);
    this.name = "PreludioError";
    this.category = category;
    this.detail = detail;
  }
}

export function fromWasmError(err: unknown): PreludioError {
  if (err instanceof PreludioError) return err;

  if (typeof err === "object" && err !== null) {
    for (const category of categories) {
      if (Object.prototype.hasOwnProperty.call(err, category)) {
        return new PreludioError(category, (err as Record<string, unknown>)[category]);
      }
    }
  }

  const message = err instanceof Error ? err.message : typeof err === "string" ? err : undefined;
  return new PreludioError("ExecutionFailed", err, message);
}
