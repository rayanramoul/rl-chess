import React from 'react';
import Board from './board.js';
export default class Game extends React.Component{
constructor(props)
{
	super(props);
	this.state={turnBlank:true,};
}

render()
{
	return (<div><p> Turn of {this.state.turnBlank?'White':'Black'}</p><Board/></div>);
}

}