import React from 'react';


export default class Square extends React.Component{
constructor(props){
	super(props);
	this.state={piece:this.props.piece,original:this.props.color,color:this.props.color,}
}
render()
{
	var img='resources/icons/'+this.state.piece+'.png';
	return (<div class={this.state.color} onClick={()=>this.handleClick()}><img src={img} alt={this.state.piece}></img></div>);
}


handleClick()
{
	if(this.state.color!=='selected')
	{
	this.setState({color:'selected'});
	}
	else
	{
		this.setState({color:this.state.original});	
	}
}
}