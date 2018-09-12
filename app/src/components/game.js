import React from 'react';
import Board from './board.js';
export default class Game extends React.Component{
constructor(props)
{
	super(props);
	this.state={turnBlank:true, squares: 


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
	return (<div id='all'><div id='title'> Turn of {this.state.turnBlank?'White':'Black'}</div><Board squares={this.state.squares}/></div>);
}

}