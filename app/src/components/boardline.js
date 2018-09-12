import React from 'react';
import '../index.css';
import Square from './square.js'
export default class BoardLine extends React.Component{
constructor(props){
	super(props);
	this.state={squares:this.props.squares,starterWhite:this.props.starterWhite,};
}

render(){
        var names = this.print();
        var imgs=[]
        for(let i=0;i<8;i++)
        {
        	imgs[i]='resources/icons/'+this.state.squares[i]+'.png';
        }
        return (
            <div class='container'>
                {names.map((name, index)=>{
                	return <Square color={name} piece={this.state.squares[index]}/>;
                  })}
            </div>
        )
    }
print(){	
	var x=[],i;
	var doum=this.state.starterWhite;
	for(i=0;i<8;i++){
	if((i%2)===0){
		let pr=(doum==='true')?'white':'black';
		x.push(pr);
		}	
	else{
		let pr=(doum==='true')?'black':'white';
		x.push(pr);
		}
	}
	return x;
}


}