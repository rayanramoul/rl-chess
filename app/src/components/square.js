import React from 'react';


export default class Square extends React.Component{
constructor(props){
	super(props);
	this.state={piece:this.props.piece,original:this.props.color,color:this.props.color,number:this.props.number,}
}
render()
{

	var img='resources/icons/'+this.props.piece+'.png';
	return (<button className={this.state.color} onClick={this.props.onClick}><img src={img} alt={this.props.piece}></img></button>);
}

/*
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
	console.log('Hey : '+this.click);
	this.click();
}*/
}