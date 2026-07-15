export type DemoRecord = {
  passed?: boolean;
  detail?: string;
  response?: string;
  confidence?: number;
  category?: string;
};

export async function fetchDemos(): Promise<Record<string, DemoRecord>> {
  const response = await fetch("http://127.0.0.1:8000/demos");
  if (!response.ok) {
    throw new Error("Failed to load demos");
  }
  return (await response.json()) as Record<string, DemoRecord>;
}

