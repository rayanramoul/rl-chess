import React from 'react';
import BoardLine from './boardline.js';

export default class Board extends React.Component{
render()
{
	return (

<div className='chess-board'>
<div className='board-line'>8 - <BoardLine starterWhite={true}/></div>
<div className='board-line'>7 - <BoardLine starterWhite={false}/></div>
<div className='board-line'>6 - <BoardLine starterWhite={true}/></div>
<div className='board-line'>5 - <BoardLine starterWhite={false}/></div>
<div className='board-line'>4 - <BoardLine starterWhite={true}/></div>
<div className='board-line'>3 - <BoardLine starterWhite={false}/></div>
<div className='board-line'>2 - <BoardLine starterWhite={true}/></div>
<div className='board-line'>1 - <BoardLine starterWhite={false}/></div>
</div>
		);
}

}