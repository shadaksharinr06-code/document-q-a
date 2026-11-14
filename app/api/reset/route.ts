import { NextRequest, NextResponse } from 'next/server';
import { vectorStore } from '@/lib/vector-store';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const { sessionId } = body;

    // Check for Vercel internal service binding: BACKEND_URL
    const backendUrl = process.env.BACKEND_URL;
    if (backendUrl) {
      try {
        const targetUrl = new URL('/api/reset', backendUrl);
        const backendRes = await fetch(targetUrl.toString(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (backendRes.ok) {
          const data = await backendRes.json();
          return NextResponse.json(data);
        }
      } catch (proxyErr) {
        console.warn('Proxy to BACKEND_URL failed, falling back to local handler:', proxyErr);
      }
    }

    if (sessionId) {
      vectorStore.clearSession(sessionId);
    }

    return NextResponse.json({
      success: true,
      message: 'Document cache and session reset successfully.',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error?.message || 'Failed to reset session' },
      { status: 500 }
    );
  }
}
