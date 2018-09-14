import React from 'react';
import Board from './board.js';
export default class Game extends React.Component{
constructor(props)
{
	super(props);
	this.state={lastSelected:[null,null],turnBlank:true, squares: 


		[
		['black_rook','black_knight','black_bishop','black_king','black_queen','black_bishop','black_knight','black_rook'],
		['black_pawn','black_pawn','black_pawn','black_pawn','black_pawn','black_pawn','black_pawn','black_pawn',],
		[null,null,null,null,null,null,null,null],
		[null,null,null,null,null,null,null,null],
		[null,null,null,null,null,null,null,null],
		[null,null,null,null,null,null,null,null],
				['white_pawn','white_pawn','white_pawn','white_pawn','white_pawn','white_pawn','white_pawn','white_pawn',],
		['white_rook','white_knight','white_bishop','white_king','white_queen','white_bishop','white_knight','white_rook']
		]
	,};
}

render()
{
	return (<div id='all'><div id='title'> Turn of {this.state.turnBlank?'White':'Black'}</div><Board squares={this.state.squares} onClick={(i, j, callback)=>this.changed(i, j)}/></div>);
}

changed(i, j)
{
	console.log('Line : '+i+' Column : '+j);
	console.log('State : ')
	let iOrigin=this.state.lastSelected[0];
	let jOrigin=this.state.lastSelected[1];
	let pionOrigin;
	if(iOrigin!==null || jOrigin!==null) {
		pionOrigin=this.state.squares[iOrigin][jOrigin];
	}
	else{
		pionOrigin=null;
	}
	let pionDestination=this.state.squares[i][j];
	if(iOrigin===i && jOrigin===j)
	{
		console.log('It was the last selected');
		this.setState({lastSelected:[null,null]});
	}
	else if(iOrigin===null || jOrigin===null)
	{
		this.setState({lastSelected:[i,j]});
	}
	else if(this.movePossible(pionOrigin, pionDestination, iOrigin, jOrigin, i, j))
	{
		let squares=this.state.squares;
		console.log('Squares : '+squares[0]);
		squares[i][j]=squares[this.state.lastSelected[0]][this.state.lastSelected[1]];
		squares[this.state.lastSelected[0]][this.state.lastSelected[1]]=null;
		this.setState({squares:squares,lastSelected:[null,null],turnBlank:!this.state.turnBlank});


	}
	else{
		this.setState({lastSelected:[null,null],});
	}
}

movePossible(pionOrigin,pionDestination,iOrigin,jOrigin,iDestination,jDestination)
{
	// DIRECTION OF MOVE FOR PAWNS
	let negate=this.turnBlank?1:-1;

	// PLAY PER TURN
	if((this.state.turnBlank && pionOrigin.startsWith('black')) || (!this.state.turnBlank && pionOrigin.startsWith('white'))){return false;}
	if(pionOrigin===null || pionDestination===null || iOrigin===null || jOrigin===null || iDestination===null || jDestination===null){return true;}
	// CAN'T EAT THE SAME COLOR

	if(( pionOrigin.startsWith('white') && pionDestination.startsWith('white')) || ( pionOrigin.startsWith('black') && pionDestination.startsWith('black')))
		{return false;}
	else
		{return true;}

	// PAWNS MOVES



	// KING MOVES


	// QUEEN MOVES



	// ROOK MOVES


	// BISHOP MOVES


	// KNIGHT MOVES




}

}