import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface ProbabilityChartProps {
  probabilities: Record<string, number> | null
}

export function ProbabilityChart({ probabilities }: ProbabilityChartProps) {
  const data = Array.from({ length: 10 }, (_, i) => ({
    digit: String(i),
    probability: probabilities ? (probabilities[String(i)] ?? 0) * 100 : 0,
  }))

  return (
    <div className="h-64 w-full" role="img" aria-label="Class probability chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis dataKey="digit" tick={{ fill: '#4b5563', fontSize: 12 }} />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: '#4b5563', fontSize: 12 }}
            tickFormatter={(v: number) => `${v}%`}
            width={40}
          />
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Probability']}
            contentStyle={{
              borderRadius: 8,
              borderColor: '#e5e7eb',
              fontSize: 13,
            }}
          />
          <Bar dataKey="probability" fill="#0f766e" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
