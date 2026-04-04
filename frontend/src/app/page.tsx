"use client";

import { useState } from "react";

interface SourceRef {
  url: string;
  title: string;
  excerpt: string;
}

interface ExtractedCell {
  value: string | null;
  confidence: number;
  sources: SourceRef[];
}

interface ColumnSpec {
  name: string;
  description: string;
  importance: "high" | "medium" | "low";
}

interface FinalRow {
  entity_id: string;
  entity: string;
  cells: Record<string, ExtractedCell>;
}

interface SearchResponse {
  topic: string;
  entity_type: string;
  columns: ColumnSpec[];
  rows: FinalRow[];
  diagnostics: Record<string, unknown>;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [data, setData] = useState<SearchResponse | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setData(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

      const response = await fetch(`${apiUrl}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: query }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(JSON.stringify(result, null, 2));
      }

      console.log("backend result", result);
      setData(result);
    } catch (error) {
      console.error(error);
      alert(error instanceof Error ? error.message : "Something went wrong!");
    } finally {
      setIsSearching(false);
    }
  };

  const prettyLabel = (key: string) =>
    key.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <main className="min-h-screen bg-gray-50 p-8 font-sans text-gray-900">
      <div className="max-w-6xl mx-auto space-y-8 mt-12">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900">
            Agentic Research Engine
          </h1>
          <p className="text-lg text-gray-600">
            Enter any topic. The system plans queries, searches the web, extracts structured fields, merges duplicates, and returns a grounded table.
          </p>
        </div>

        <form onSubmit={handleSearch} className="flex gap-4 max-w-2xl mx-auto">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., top AI startups in healthcare, best winter jackets under $200, or open-source vector databases"
            className="flex-1 rounded-lg border border-gray-300 px-4 py-3 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
          />
          <button
            type="submit"
            disabled={isSearching}
            className="rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white shadow-sm hover:bg-blue-500 disabled:opacity-50 transition-all"
          >
            {isSearching ? "Searching..." : "Search"}
          </button>
        </form>

        {data && (
          <div className="space-y-6">
            <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-4 text-left font-semibold text-gray-900">
                      Entity
                    </th>
                    {data.columns
                      .filter((c) => c.name !== "entity")
                      .map((col) => (
                        <th
                          key={col.name}
                          className="px-6 py-4 text-left font-semibold text-gray-900"
                        >
                          {prettyLabel(col.name)}
                        </th>
                      ))}
                  </tr>
                </thead>

                <tbody className="divide-y divide-gray-200 bg-white">
                  {data.rows.map((row) => (
                    <tr key={row.entity_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">
                        {row.entity}
                      </td>

                      {data.columns
                        .filter((c) => c.name !== "entity")
                        .map((col) => {
                          const cell = row.cells[col.name];
                          const value = cell?.value ?? "N/A";
                          const firstSource = cell?.sources?.[0];

                          return (
                            <td key={col.name} className="px-6 py-4 text-gray-600 align-top">
                              <div>{value}</div>
                              {firstSource && (
                                <a
                                  href={firstSource.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-500 hover:underline text-xs mt-1 inline-block"
                                >
                                  Source
                                </a>
                              )}
                            </td>
                          );
                        })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}