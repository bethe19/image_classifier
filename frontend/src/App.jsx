import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8003'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please select an image file')
        return
      }
      setFile(selectedFile)
      setError(null)
      setResult(null)
      
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result)
      }
      reader.readAsDataURL(selectedFile)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select an image')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while classifying the image')
    } finally {
      setLoading(false)
    }
  }

  const clearResults = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-orange-600'
    if (confidence >= 0.6) return 'text-orange-500'
    return 'text-black'
  }

  return (
    <div className="min-h-screen bg-white font-grotesk">
      <div className="container mx-auto px-4 py-12 max-w-5xl">
        <header className="text-center mb-12 border-b-2 border-black pb-8">
          <h1 className="text-5xl font-bold text-black mb-4 tracking-tight">
            HORSE & DONKEY CLASSIFIER
          </h1>
          <p className="text-lg text-black font-medium">
            Classify images using CNN deep learning
          </p>
        </header>

        <div className="bg-white border-2 border-black p-8 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="image" className="block text-sm font-bold text-black mb-3 uppercase tracking-wide">
                Upload Image
              </label>
              <input
                ref={fileInputRef}
                id="image"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="w-full px-4 py-3 border-2 border-black bg-white text-black focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 font-grotesk"
              />
            </div>

            {preview && (
              <div className="border-2 border-black p-4">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-w-full h-auto max-h-96 mx-auto"
                />
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !file}
              className="w-full bg-orange-600 text-white py-4 px-6 font-bold text-sm uppercase tracking-wider hover:bg-orange-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors border-2 border-black"
            >
              {loading ? 'Classifying...' : 'Classify Image'}
            </button>
          </form>
        </div>

        {error && (
          <div className="bg-white border-2 border-black p-4 mb-8">
            <div className="flex items-center">
              <span className="text-black mr-3 font-bold">⚠</span>
              <p className="text-black font-medium">{error}</p>
            </div>
          </div>
        )}

        {result && (
          <div className="space-y-6">
            <div className="bg-white border-2 border-black p-8">
              <div className="flex items-center justify-between mb-6 border-b-2 border-black pb-4">
                <h2 className="text-3xl font-bold text-black uppercase tracking-tight">Classification Result</h2>
                <span className="text-4xl text-orange-600">✓</span>
              </div>
              
              <div className="space-y-4">
                {preview && (
                  <div className="border-2 border-black p-4 mb-4">
                    <img
                      src={preview}
                      alt="Classified"
                      className="max-w-full h-auto max-h-96 mx-auto"
                    />
                  </div>
                )}
                
                <div className="flex items-center justify-between pt-4 border-t-2 border-black">
                  <div>
                    <p className="text-xs font-bold text-black mb-2 uppercase tracking-wider">Prediction:</p>
                    <p className="text-2xl font-bold uppercase tracking-wide text-orange-600">
                      {result.prediction}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-bold text-black mb-2 uppercase tracking-wider">Confidence:</p>
                    <p className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {result.probabilities && (
                  <div className="pt-4 border-t-2 border-black">
                    <p className="text-xs font-bold text-black mb-3 uppercase tracking-wider">Probabilities:</p>
                    <div className="space-y-2">
                      {Object.entries(result.probabilities).map(([className, prob]) => (
                        <div key={className} className="flex items-center justify-between">
                          <span className="text-sm font-medium text-black uppercase">{className}:</span>
                          <span className={`text-sm font-bold ${getConfidenceColor(prob)}`}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            <button
              onClick={clearResults}
              className="w-full bg-white text-black py-3 px-4 font-bold text-sm uppercase tracking-wider hover:bg-black hover:text-white transition-colors border-2 border-black"
            >
              Clear Results
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

