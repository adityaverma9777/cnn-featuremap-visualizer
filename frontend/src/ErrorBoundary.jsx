import React, { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, errorMessage: "" };
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      errorMessage: error?.message || "Unexpected frontend error.",
    };
  }

  componentDidCatch(error) {
    console.error("UI crash captured by ErrorBoundary:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main style={{ padding: "24px", fontFamily: "Segoe UI, sans-serif", color: "#222" }}>
          <h1>UI Error</h1>
          <p>The app encountered an error while rendering.</p>
          <p>
            <strong>Details:</strong> {this.state.errorMessage}
          </p>
          <p>Refresh the page after saving changes.</p>
        </main>
      );
    }

    return this.props.children;
  }
}
