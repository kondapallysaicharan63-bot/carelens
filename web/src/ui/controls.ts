// Binds a control to a session that may only exist later: the handler is registered AND immediately seeded with the
// control's current value, so a change the user made while the model was loading is not lost. DOM-free (testable).
export function seedAndSubscribe<T>(read: () => T, subscribe: (onChange: () => void) => void, apply: (v: T) => void): void {
  subscribe(() => apply(read()));
  apply(read());
}
