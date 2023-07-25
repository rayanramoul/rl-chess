import './Chessboard.scss';
import Tile from '../Tile/Tile';

const horizontalAxis = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const verticalAxis = ['1', '2', '3', '4', '5', '6', '7', '8'];

interface Piece {
  image: string;
  x: number;
  y: number;
  number: number;
}

const pieces: Piece[] = [];
const order_of_pieces = ["R", "N", "B", "Q", "K", "B", "N", "R"]
// pieces.push({ image: "assets/bB.png", x: 0, y: 0, number: 0 });

for (let i = 0; i < 8; i++) {
    pieces.push({ image: "assets/b"+order_of_pieces[i]+".png", x: i, y: 0, number: i });
}

for (let i = 0; i < 8; i++) {
  pieces.push({ image: "assets/bP.png", x: i, y: 1, number: i+1 });
}

for (let j = 2; j < verticalAxis.length-2; j++) {
  for (let i = horizontalAxis.length - 1; i >= 0; i--) {
    pieces.push({ image: '', x: i, y: j, number: i + j + 1 });
  }
}

for (let i = 0; i < 8; i++) {
  pieces.push({ image: "assets/wP.png", x: i, y: 6, number: i + 6 });
}

for (let i = 0; i < 8; i++) {
    pieces.push({ image: "assets/w"+order_of_pieces[i]+".png", x: i, y: 7, number: i+7 });
}


function Chessboard() {
  let board = [];
  pieces.forEach((piece) => {
    board.push(<Tile key={`${piece.x},${piece.y}`} x={piece.x} y={piece.y} number={piece.number} image={piece.image} />);
  });

  return (
    <div id="chessboard">
      {board}
    </div>
  );
}

export default Chessboard;
