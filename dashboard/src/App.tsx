import { useEffect, useState, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Activity, TrendingUp, TrendingDown, Layers, Crosshair, Zap, ShieldAlert } from 'lucide-react';

interface TelemetryData {
  timestamp: number;
  alpha: number;
  vpin: number;
  book_imbalance: number;
  volatility: number;
  regime: number;
  best_bid: number;
  best_ask: number;
  equity: number;
  inventory: number;
  cash: number;
}

const REGIME_MAP = ["NORMAL", "HIGH_TOXICITY", "LOW_LIQUIDITY", "TRENDING", "UNKNOWN"];

function App() {
  const [data, setData] = useState<TelemetryData[]>([]);
  const [connected, setConnected] = useState(false);
  const [isTrading, setIsTrading] = useState(false);
  const [currentAlpha, setCurrentAlpha] = useState(0);
  
  // Simulated paper trading state since we don't have full portfolio state from engine yet
  const [mockEquity, setMockEquity] = useState(10000000);
  const [inventory, setInventory] = useState(0); // BTC

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to FastAPI Telemetry Server
    const connectWs = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/telemetry');
      
      ws.onopen = () => setConnected(true);
      
      ws.onmessage = (event) => {
        const payload: TelemetryData = JSON.parse(event.data);
        setCurrentAlpha(payload.alpha);

        const mtmEquity = payload.equity; // Backend computes limit order equity

        setMockEquity(mtmEquity);
        setInventory(payload.inventory);

        setData(prev => {
          const updated = [...prev, { ...payload, equity: mtmEquity }].slice(-50);
          return updated;
        });
      };

      ws.onclose = () => {
        setConnected(false);
        setTimeout(connectWs, 2000); // Reconnect
      };
      
      wsRef.current = ws;
    };
    
    connectWs();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Fetch initial trading status
  useEffect(() => {
    fetch('http://localhost:8000/api/trade_status')
      .then(res => res.json())
      .then(data => setIsTrading(data.is_trading))
      .catch(console.error);
  }, []);

  const toggleTrading = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/trade_control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_trading: !isTrading })
      });
      const data = await res.json();
      setIsTrading(data.is_trading);
    } catch (e) {
      console.error(e);
    }
  };

  const latest = data[data.length - 1] || null;

  return (
    <div className="min-h-screen bg-background text-gray-200 p-6 flex flex-col gap-6">
      
      {/* HEADER */}
      <header className="flex justify-between items-center pb-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Zap className="text-primary w-8 h-8" />
          <h1 className="text-2xl font-bold tracking-tight text-white">Quant Cockpit</h1>
          <span className="bg-primary/20 text-primary px-3 py-1 rounded-full text-xs font-semibold uppercase ml-2">
            Paper Trading
          </span>
        </div>
        <div className="flex items-center gap-4">
          <button 
            onClick={toggleTrading}
            className={`px-4 py-2 rounded font-bold transition-colors ${
              isTrading 
                ? 'bg-danger/20 text-danger hover:bg-danger/30 border border-danger/50' 
                : 'bg-success/20 text-success hover:bg-success/30 border border-success/50'
            }`}
          >
            {isTrading ? 'STOP TRADING' : 'START TRADING'}
          </button>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${connected ? 'bg-success animate-pulse' : 'bg-danger'}`} />
            <span className="text-sm font-medium text-gray-400">
              {connected ? 'Binance Live (ws://)' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* TOP STATS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        
        {/* Equity */}
        <div className="glass-panel p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity className="w-16 h-16" />
          </div>
          <div className="data-label mb-2 flex items-center gap-2">
            Paper Equity (USD)
          </div>
          <div className="data-value text-3xl text-white">
            ${mockEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        {/* Alpha Signal */}
        <div className="glass-panel p-5">
          <div className="data-label mb-2 flex items-center gap-2">
            <Crosshair className="w-4 h-4 text-primary" />
            ML Alpha Signal
          </div>
          <div className={`data-value text-3xl flex items-center gap-2 ${currentAlpha > 0.1 ? 'text-success' : currentAlpha < -0.1 ? 'text-danger' : 'text-white'}`}>
            {currentAlpha > 0.1 ? <TrendingUp /> : currentAlpha < -0.1 ? <TrendingDown /> : null}
            {currentAlpha.toFixed(4)}
          </div>
        </div>

        {/* Spread */}
        <div className="glass-panel p-5">
          <div className="data-label mb-2 flex items-center gap-2">
            <Layers className="w-4 h-4 text-warning" />
            Live Market (BTCUSDT)
          </div>
          <div className="flex justify-between items-end">
            <div>
              <div className="text-xs text-gray-500 font-mono">BID</div>
              <div className="data-value text-success text-xl">{latest?.best_bid?.toFixed(2) || '---'}</div>
            </div>
            <div className="text-gray-600 mb-1 font-mono text-sm">
              {(latest?.best_ask - latest?.best_bid)?.toFixed(2) || '-'}
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500 font-mono">ASK</div>
              <div className="data-value text-danger text-xl">{latest?.best_ask?.toFixed(2) || '---'}</div>
            </div>
          </div>
        </div>

        {/* Regime */}
        <div className="glass-panel p-5 border border-danger/30 bg-danger/5">
          <div className="data-label mb-2 flex items-center gap-2 text-danger">
            <ShieldAlert className="w-4 h-4" />
            Market Regime
          </div>
          <div className="data-value text-2xl text-danger/90">
            {latest ? REGIME_MAP[latest.regime] : '---'}
          </div>
        </div>
      </div>

      {/* CHARTS SECTION */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-grow">
        
        {/* Main Chart */}
        <div className="glass-panel p-5 lg:col-span-2 flex flex-col">
          <h2 className="text-lg font-semibold text-white mb-6">Equity Curve (Live)</h2>
          <div className="flex-grow min-h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 5, right: 0, left: 20, bottom: 5 }}>
                <XAxis dataKey="timestamp" hide />
                <YAxis 
                  domain={['auto', 'auto']} 
                  tick={{ fill: '#8b949e', fontSize: 12, fontFamily: 'monospace' }}
                  tickFormatter={(val) => `$${(val / 1000000).toFixed(2)}M`}
                  orientation="right"
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#18181b', borderColor: '#30363d', color: '#fff', borderRadius: '8px' }}
                  itemStyle={{ fontFamily: 'monospace', color: '#3b82f6' }}
                  formatter={(value: number) => [`$${value.toFixed(2)}`, 'Equity']}
                  labelFormatter={() => ''}
                />
                <Line 
                  type="monotone" 
                  dataKey="equity" 
                  stroke="#3b82f6" 
                  strokeWidth={2} 
                  dot={false}
                  isAnimationActive={false} // Disable for perf on live data
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Alpha Features */}
        <div className="glass-panel p-5 flex flex-col gap-4">
          <h2 className="text-lg font-semibold text-white mb-2">Microstructure Features</h2>
          
          <div className="bg-surface p-4 rounded-lg border border-white/5">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-gray-400">Order Book Imbalance</span>
              <span className="font-mono text-sm">{latest?.book_imbalance?.toFixed(3) || '0.000'}</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2 overflow-hidden flex">
              {/* OBI ranges from -1 to 1. Map to 0-100% split */}
              <div 
                className="bg-danger h-full transition-all duration-300" 
                style={{ width: `${Math.max(0, 50 - (latest?.book_imbalance || 0) * 50)}%` }}
              />
              <div 
                className="bg-success h-full transition-all duration-300"
                style={{ width: `${Math.max(0, 50 + (latest?.book_imbalance || 0) * 50)}%` }}
              />
            </div>
          </div>

          <div className="bg-surface p-4 rounded-lg border border-white/5">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-gray-400">VPIN (Toxicity)</span>
              <span className="font-mono text-sm">{latest?.vpin?.toFixed(4) || '0.0000'}</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2">
              <div className="bg-warning h-1.5 rounded-full transition-all duration-300" style={{ width: `${(latest?.vpin || 0) * 100}%` }}></div>
            </div>
          </div>
          
          <div className="bg-surface p-4 rounded-lg border border-white/5">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-gray-400">Realized Volatility</span>
              <span className="font-mono text-sm">{latest?.volatility?.toFixed(2) || '0.00'}</span>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}

export default App;
