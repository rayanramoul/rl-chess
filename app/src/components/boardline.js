import React from 'react';

export default class BoardLine extends React.Component{
constructor(props){
	super(props);
	this.state={squares:Array(8).fill(null),starterWhite:this.props.starterWhite,};
}

	render(){
const d=this.print();
		return (<p>{d}</p>);
}


print(){	
	var x='',i;
	var doum=this.state.starterWhite;
	for(i=0;i<8;i++){
	if((i%2)===0){
		let pr=(doum=='true')?' | White | ':' | Black | ';
		x=x+pr;
		}	
	else{
		let pr=(doum=='true')?' | Black | '	:' | White | ';
		x=x+pr;
		}
	}
	return x;
}


}
