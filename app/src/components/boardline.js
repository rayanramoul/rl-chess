import React from 'react';

export default class BoardLine extends React.Component{
constructor(props){
	super(props);
	this.state={squares:Array(8).fill(null),starterWhite:this.props.starterWhite,};
}

	render(){
const d=this.print();
		return (<p>This is {d}</p>);
}


print(){	
	var x='',i; 
	for(i=0;i<8;i++){
	if((i%2)===0){
		x=x+' | White | ';
		}	
	else{
		x=x+' | Black | ';
		}
	}
	return x;
}


}
