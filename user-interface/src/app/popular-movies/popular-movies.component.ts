import { Component, OnInit } from '@angular/core';
import { Movie, MoviesService } from '../movies.service';

@Component({
  selector: 'app-popular-movies',
  templateUrl: './popular-movies.component.html',
  styleUrls: ['./popular-movies.component.css']
})
export class PopularMoviesComponent implements OnInit {

  movies!: Movie[];


  constructor(private moviesService: MoviesService) { }

  ngOnInit(): void {

    this._getAllTeams();
    //getPopularMovies(): Promise<Movie>

  }

  _getAllTeams():void{
    this.moviesService.getPopularMovies()
      .then((movies)=>this._fillTeams(movies))
      .catch((error)=>this._handleError(error))
  }

  
  _fillTeams(movies: Movie[]){
    this.movies = movies;
    //console.log("_fillTeams: "+ JSON.stringify(this.movies));
    return this.movies;
  }

  
  _handleError(error: any){
    console.log(error);
    alert(error.message)

  }
}
