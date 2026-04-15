import React from 'react';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[VoiceEye] Unhandled error:', error, info.componentStack);

    // Speak the error for visually impaired users
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(
        'VoiceEye encountered an error. Please tap the screen to reload.'
      );
      u.rate = 1.0;
      window.speechSynthesis.speak(u);
    }
  }

  private handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="error-boundary"
          onClick={this.handleReload}
          role="alert"
          style={{
            position: 'fixed',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0f172a',
            color: '#f1f5f9',
            padding: '2rem',
            textAlign: 'center',
            fontFamily: 'system-ui, sans-serif',
            cursor: 'pointer',
          }}
        >
          <h1 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>
            Something went wrong
          </h1>
          <p style={{ fontSize: '1rem', opacity: 0.7, marginBottom: '2rem', maxWidth: '400px' }}>
            VoiceEye encountered an unexpected error. Tap anywhere to reload.
          </p>
          <button
            onClick={this.handleReload}
            aria-label="Reload application"
            style={{
              padding: '0.75rem 2rem',
              fontSize: '1rem',
              borderRadius: '0.5rem',
              border: '1px solid rgba(255,255,255,0.2)',
              background: 'rgba(99, 102, 241, 0.3)',
              color: '#f1f5f9',
              cursor: 'pointer',
            }}
          >
            Reload
          </button>
          {this.state.error && (
            <pre
              style={{
                marginTop: '2rem',
                fontSize: '0.75rem',
                opacity: 0.4,
                maxWidth: '90vw',
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
              }}
            >
              {this.state.error.message}
            </pre>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
