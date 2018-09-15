import React from 'react';
import Board from './board.js';
export default class Game extends React.Component{
constructor(props)
{
	super(props);
	this.state={lastSelected:["nothing","nothing"],turnBlank:true, squares: 


		[
		['black_rook','black_knight','black_bishop','black_king','black_queen','black_bishop','black_knight','black_rook'],
		['black_pawn','black_pawn','black_pawn','black_pawn','black_pawn','black_pawn','black_pawn','black_pawn',],
		["nothing","nothing","nothing","nothing","nothing","nothing","nothing","nothing"],
		["nothing","nothing","nothing","nothing","nothing","nothing","nothing","nothing"],
		["nothing","nothing","nothing","nothing","nothing","nothing","nothing","nothing"],
		["nothing","nothing","nothing","nothing","nothing","nothing","nothing","nothing"],
				['white_pawn','white_pawn','white_pawn','white_pawn','white_pawn','white_pawn','white_pawn','white_pawn',],
		['white_rook','white_knight','white_bishop','white_queen','white_king','white_bishop','white_knight','white_rook']
		]
	,};
}

render()
{
	return (<div id='all'><div id='title'> Turn of {this.state.turnBlank?'White':'Black'}</div><Board squares={this.state.squares} onClick={(i, j, callback)=>this.changed(i, j)}/></div>);
}

changed(i, j)
{
	let iOrigin=this.state.lastSelected[0];
	let jOrigin=this.state.lastSelected[1];
	let pionOrigin;
	let pionDestination=this.state.squares[i][j];
	console.log(pionOrigin, pionDestination, iOrigin, jOrigin, i, j)


	if(iOrigin!=="nothing" || jOrigin!=="nothing") {
		pionOrigin=this.state.squares[iOrigin][jOrigin];
	}
	else{
		pionOrigin="nothing";
	}
	

	if(iOrigin===i && jOrigin===j)
	{
		console.log('It was the last selected');
		this.setState({lastSelected:["nothing","nothing"]});
	}
	else if(iOrigin==="nothing" || jOrigin==="nothing")
	{
		this.setState({lastSelected:[i,j]});
	}
	else if(this.movePossible(pionOrigin, pionDestination, iOrigin, jOrigin, i, j))
	{
		let squares=this.state.squares;
		console.log('Squares : '+squares[0]);
		squares[i][j]=squares[this.state.lastSelected[0]][this.state.lastSelected[1]];
		squares[this.state.lastSelected[0]][this.state.lastSelected[1]]="nothing";
		this.setState({squares:squares,lastSelected:["nothing","nothing"],turnBlank:!this.state.turnBlank});


	}
	else{
		this.setState({lastSelected:["nothing","nothing"],});
	}
}

movePossible(pionOrigin,pionDestination,iOrigin,jOrigin,iDestination,jDestination)
{

	// DIRECTION OF MOVE FOR PAWNS
	let negate=this.turnBlank?1:-1;
	// PLAY PER TURN
	if((this.state.turnBlank && pionOrigin.startsWith('black')) || (!this.state.turnBlank && pionOrigin.startsWith('white'))){return false;}

	// CAN'T EAT THE SAME COLOR

	if(( pionOrigin.startsWith('white') && pionDestination.startsWith('white')) || ( pionOrigin.startsWith('black') && pionDestination.startsWith('black')))
		{return false;}


	// PAWNS MOVES
	if(pionOrigin.endsWith('pawn'))
	{
		if((iOrigin===1) && this.turnBlank){
			if(jOrigin===jDestination && iOrigin-iDestination>-3){return true;}
			else {return false;}
		}
		else if(iOrigin===6 && !this.turnBlank){
			if(jOrigin===jDestination && iOrigin-iDestination<3){return true;}
			else {return false;}
		}
		else{
			return false;
		}

	}


	// KING MOVES
 if((pionOrigin.endsWith('king')))
 {
 	if(iOrigin-iDestination>1 || iOrigin-iDestination<-1 || jOrigin-jDestination>1 || jOrigin-jDestination<-1 ) return false;
	return true;
 }
	// QUEEN MOVES

 if((pionOrigin.endsWith('queen')))
 {
 	if(iOrigin===iDestination && jDestination!==jOrigin) return true;
 	else if(jOrigin===jDestination && iDestination!==iOrigin) return true;
 	else if(((iOrigin-iDestination)===(jOrigin-jDestination)) || ((iOrigin-iDestination)===-1*(jOrigin-jDestination))) return true;
 	else return false;
 }


	// ROOK MOVES
 if((pionOrigin.endsWith('rook')))
 {
 	if(iOrigin===iDestination && jDestination!==jOrigin) return true;
 	else if(jOrigin===jDestination && iDestination!==iOrigin) return true;
 	else return false;
 }



	// BISHOP MOVES

 if((pionOrigin.endsWith('bishop')))
 {
 	if(((iOrigin-iDestination)===(jOrigin-jDestination)) || ((iOrigin-iDestination)===-1*(jOrigin-jDestination))) return true;
 	else return false;
 }


	// KNIGHT MOVES

if(pionOrigin.endsWith('knight'))
{

}


}

}