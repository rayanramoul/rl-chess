import React from 'react';
import BoardLine from './boardline.js';
import '../index.css';
export default class Board extends React.Component{
	constructor(props){
		super(props);
		this.props=props;
		this.state={squares:this.props.squares,};
	}
render()
{
	return (

<div className='chess-board'>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[0]} onClick={(i)=>this.props.onClick(0, i)}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[1]} onClick={(i)=>this.props.onClick(1, i)}/>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[2]} onClick={(i)=>this.props.onClick(2, i)}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[3]} onClick={(i)=>this.props.onClick(3, i)}/>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[4]} onClick={(i)=>this.props.onClick(4, i)}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[5]} onClick={(i)=>this.props.onClick(5, i)}/>
<BoardLine className='board-line' starterWhite='true' squares={this.state.squares[6]} onClick={(i)=>this.props.onClick(6, i)}/>
<BoardLine className='board-line' starterWhite='false' squares={this.state.squares[7]} onClick={(i)=>this.props.onClick(7, i)}/>
</div>
		);
}

}