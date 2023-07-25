import './Tile.scss'

interface Props {
    image: string,
    number: number
    x?: string,
    y?: string
}

export default function Tile({number, image, x = '', y = ''}: Props)
{
    const piece = (image === '') ? '' : (<img className="chess-piece-image" src={image}/>)
    if((number)%2==0)
    {   return (<span className="tile black-tile">{piece}{x}{y}</span>); }
    else
    { return (<span className="tile white-tile">{piece}{x}{y}</span>); }
}