import { render, screen } from "@testing-library/react";

import App from "./App";

describe("App", () => {
  it("renders the Forge dashboard title", () => {
    render(<App />);
    expect(screen.getByText(/cognitive behaviors, traced and testable/i)).toBeInTheDocument();
  });
});

