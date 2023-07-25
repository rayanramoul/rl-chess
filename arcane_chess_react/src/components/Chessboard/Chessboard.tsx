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

function grabPiece(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    console.log(e.target);
    const target = e.target as HTMLDivElement;
    if (target.classList.contains('chess-piece-image')) {
        target.style.position = 'absolute';
        target.style.zIndex = '1000';
        const shiftX = e.clientX - target.getBoundingClientRect().left;
        const shiftY = e.clientY - target.getBoundingClientRect().top;
    
        moveAt(e.pageX, e.pageY);
    
        function moveAt(pageX: number, pageY: number) {
        target.style.left = pageX - shiftX + 'px';
        target.style.top = pageY - shiftY + 'px';
        }
    
        function onMouseMove(e: MouseEvent) {
        moveAt(e.pageX, e.pageY);
        }
    
        document.addEventListener('mousemove', onMouseMove);
    
        target.onmouseup = function () {
        document.removeEventListener('mousemove', onMouseMove);
        target.onmouseup = null;
        };
    }
}


function Chessboard() {
  let board = [];
  pieces.forEach((piece) => {
    board.push(<Tile key={`${piece.x},${piece.y}`} x={piece.x} y={piece.y} number={piece.number} image={piece.image} />);
  });



  return (
    <div id="chessboard" onMouseDown={e => grabPiece(e)}>
      {board}
    </div>
  );
}

export default Chessboard;
