# AGENT.md

## FastAPI Guidelines

This is a small FastAPI application. Keep it **simple, accurate, documented, and reasonably secure**.

### Structure

* Keep all FastAPI code in `main.py`.
* Keep reusable low-level utilities in `modules/`.
* SQLite access belongs in a small class/module under `modules/`.
* Do not introduce routers, services, repositories, or other layers unless genuinely needed.

### Code

* Prefer straightforward code over abstractions and design patterns.
* Use type hints and Pydantic models.
* Follow the existing code style and structure.
* Do not refactor unrelated code.
* Do not guess about application behavior; inspect the existing code first.
* Avoid unnecessary dependencies.

### API Documentation

Use FastAPI's built-in OpenAPI/Swagger documentation.

* Give every endpoint a clear `summary` and useful `description`.
* Use clear endpoint names and parameter names.
* Add descriptions to important request/response fields.
* Define request and response models with Pydantic.
* Use appropriate HTTP status codes and document important responses.
* Document authentication requirements where applicable.
* Keep the Swagger documentation accurate whenever an endpoint changes.
* Do not create separate API documentation unless explicitly requested.

Example:

```python
@app.get(
    "/users/{user_id}",
    summary="Get a user",
    description="Returns a user by their ID.",
    response_model=UserResponse,
)
def get_user(user_id: int):
    ...
```

Keep documentation **short and useful**. It should help a developer understand how to use the endpoint without explaining obvious implementation details.

### Security

* Validate untrusted input.
* Use parameterized SQL queries; never build SQL with user input.
* Check authentication and authorization on protected endpoints.
* Check resource ownership where applicable.
* Never hard-code or log secrets, tokens, passwords, or sensitive data.
* Do not expose internal errors, stack traces, SQL, or filesystem details to clients.
* Do not blindly fetch user-provided URLs.

### SQLite

* Use the SQLite helper from `modules/` for database access.
* Handle connections and transactions safely.
* Keep SQL simple and readable.
* Do not build an ORM-like abstraction around SQLite.

### General Rule

**Write the simplest code that is correct, maintainable, well-documented through FastAPI's Swagger/OpenAPI, and reasonably secure. Do not over-engineer.**
