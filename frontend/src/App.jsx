import { useState } from 'react'
import { jsPDF } from "jspdf";

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null) 

  const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";

  const isUrl = text.trim().startsWith('http://') || text.trim().startsWith('https://');

  const handleSummarize = async () => {
    if (!text) return
    setLoading(true)
    setErrorMsg(null) // Clear past errors before initiating a run
    setResult(null)   // Clear previous results for clean state transition
    
    try {
      const endpoint = isUrl ? `${API_URL}/scrape` : `${API_URL}/generate`;
      const bodyPayload = isUrl ? { url: text.trim() } : { text };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyPayload)
      })
      
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || "System processing failure encountered.");
      }

      setResult(data)
    } catch (error) {
      console.error("NewsScribe error:", error)
      setErrorMsg(error.message || "Unable to communicate with the processing engine.")
    } finally {
      setLoading(false)
    }
  }

  const exportToPDF = () => {
    if (!result) return;
    const doc = new jsPDF();
    const date = new Date().toLocaleDateString();
    const cap = (str) => str ? str.charAt(0).toUpperCase() + str.slice(1) : "";

    doc.setFont("serif", "bold");
    doc.setFontSize(22);
    doc.text("NewsScribe — Summary Report", 105, 20, { align: "center" });
    doc.setFontSize(10);
    doc.text(`Generated: ${date}`, 20, 38);
    doc.line(20, 32, 190, 32);

    doc.setFontSize(14);
    doc.setFont("serif", "bold");
    doc.text("Generated Summary:", 20, 45);
    doc.setFont("serif", "normal");
    doc.setFontSize(12);
    const summaryLines = doc.splitTextToSize(cap(result.summary), 170);
    doc.text(summaryLines, 20, 52);

    const footerY = 60 + summaryLines.length * 6;
    doc.line(20, footerY, 190, footerY);
    
    // Using optional chaining to make the PDF exporter fully bulletproof
    const sentiment = result?.metadata?.sentiment || "N/A";
    const score = result?.metadata?.score ? Math.round(result.metadata.score * 100) : 0;
    const latency = result?.metadata?.latency_ms || "N/A";
    const device = result?.metadata?.device || "CPU";

    doc.text(`Sentiment: ${sentiment} (${score}%)`, 20, footerY + 8);
    doc.text(`Inference Latency: ${latency}ms`, 20, footerY + 14);
    doc.text(`Hardware: ${device.toUpperCase()}`, 20, footerY + 20);
    doc.save("NewsScribe_Summary_Report.pdf");
  };

  const formatHeadline = (str) => {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  const ScribeLoader = () => (
    <div className="flex flex-col items-center justify-center space-y-4 py-12">
      <div className="relative">
        <div className="w-4 h-4 bg-quill rounded-full blur-sm animate-flicker" />
        <div className="absolute top-0 w-4 h-4 bg-sepia/20 rounded-full animate-ping" />
      </div>
      <p className="text-xs tracking-[0.4em] uppercase text-sepia/40 animate-pulse">
        {isUrl ? 'Scraping and analyzing source...' : 'Analyzing text layout...'}
      </p>
    </div>
  )

  return (
    <div className="min-h-screen px-6 py-12 md:p-24 max-w-5xl mx-auto space-y-10 md:space-y-16">
      
      {/* Header */}
      <header className="text-center space-y-3">
        <h1 className="text-3xl md:text-6xl font-serif tracking-[0.2em] text-ink uppercase transition-all duration-700">
          NewsScribe
        </h1>
        <p className="text-[10px] md:text-xs font-sans tracking-[0.3em] text-sepia/60 uppercase italic">
          Automated Data Scraper & Real-Time Summarization
        </p>
      </header>

      {/* Input Field */}
      <section className="space-y-6">
        <textarea
          className="w-full h-48 md:h-80 p-6 md:p-10 bg-white/40 border border-sepia/10 rounded-sm font-serif text-base md:text-xl leading-relaxed focus:outline-none focus:border-quill/30 transition-all resize-none placeholder:italic placeholder:opacity-30"
          placeholder="Paste a direct news article text or pull live content instantly by pasting a URL link..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <div className="flex justify-center">
          <button
            onClick={handleSummarize}
            disabled={loading || !text}
            className="w-full md:w-auto px-16 py-5 border border-ink text-ink hover:bg-ink hover:text-parchment transition-all duration-500 uppercase tracking-[0.2em] text-[10px] md:text-xs disabled:opacity-20"
          >
            {loading ? 'Processing System...' : isUrl ? 'Scrape & Summarize Link' : 'Summarize Text Input'}
          </button>
        </div>
      </section>

      {/* Error Presentation Module */}
      {errorMsg && (
        <div className="p-6 border border-red-900/20 bg-red-50/10 text-center space-y-2 animate-in fade-in duration-500">
          <p className="text-xs font-sans tracking-widest text-red-700 uppercase font-bold">Execution Error</p>
          <p className="text-sm font-serif italic text-ink/80">"{errorMsg}"</p>
        </div>
      )}

      {/* Results Delivery Output */}
      {loading ? (
        <ScribeLoader />
      ) : (
        result && result.summary && (
          <div className="space-y-12 animate-in fade-in duration-1000">
            
            <ResultCard
              title="AI Generated Summary"
              text={formatHeadline(result.summary)}
              icon="📰"
              onExport={exportToPDF}
            />

            {/* Performance Audit Footer using safe optional chaining */}
            <div className="mt-12 pt-8 border-t border-sepia/10 flex flex-wrap justify-center gap-8 opacity-40 text-[10px] uppercase tracking-widest font-sans">
              <div>Inference Latency: {result?.metadata?.latency_ms || 0}ms</div>
              <div>System Engine: {(result?.metadata?.device || 'CPU').toUpperCase()}</div>
              {result?.metadata?.sentiment && (
                <div className={`font-bold ${result.metadata.sentiment === 'POSITIVE' ? 'text-green-600' : 'text-red-600'}`}>
                  Analysis Metric: {result.metadata.sentiment} ({Math.round((result?.metadata?.score || 0) * 100)}%)
                </div>
              )}
              <div>Transformer Layer: {result?.metadata?.model || 't5-custom'}</div>
            </div>
          </div>
        )
      )}
    </div>
  );
}

function ResultCard({ title, text, icon, onExport }) {
  return (
    <div className="p-8 border border-sepia/5 bg-white/20 backdrop-blur-md space-y-6 hover:border-sepia/20 transition-all group flex flex-col justify-between relative">
      <div className="space-y-4">
        
        <div className="flex justify-between items-center border-b border-sepia/10 pb-4">
          <span className="text-[9px] tracking-widest uppercase text-sepia/50 font-sans">{title}</span>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={onExport}
              className="px-3 py-1 border border-quill/40 text-quill hover:bg-quill hover:text-white transition-all text-[9px] uppercase tracking-wider font-sans rounded-xs"
            >
              📥 Export PDF
            </button>
            <span className="text-lg opacity-30 group-hover:opacity-100 transition-opacity grayscale group-hover:grayscale-0">{icon}</span>
          </div>
        </div>

        <p className="text-lg md:text-2xl font-serif italic text-ink leading-tight">"{text}"</p>
      </div>

      <button
        onClick={() => {
          navigator.clipboard.writeText(text);
          alert("Summary successfully mapped to clipboard!");
        }}
        className="text-[10px] uppercase tracking-tighter text-sepia/40 hover:text-quill transition-colors text-left pt-4 self-start"
      >
        Copy Summary Text
      </button>
    </div>
  )
}

export default App