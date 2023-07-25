import { useState } from 'react'
import Chessboard from './components/Chessboard/Chessboard'


function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="App">
      Chess Game
      <Chessboard />
      </div>
  )
}

export default App
