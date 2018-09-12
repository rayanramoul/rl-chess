import React from 'react';
import BoardLine from './boardline.js';
import '../index.css';
export default class Board extends React.Component{
	constructor(props){
		super(props);
		this.state={squares:this.props.squares,};
	}
render()
{
	return (

<div className='chess-board'>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[0]}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[1]}/>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[2]}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[3]}/>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[4]}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[5]}/>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[6]}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[7]}/>
</div>
		);
}

}