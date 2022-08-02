import { Component, OnInit } from '@angular/core';
import { Movie, MoviesService } from '../movies.service';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-similar-movies',
  templateUrl: './similar-movies.component.html',
  styleUrls: ['./similar-movies.component.css']
})
export class SimilarMoviesComponent implements OnInit {
  movies!: Movie[];
  movie: String = "";

  constructor(private moviesService: MoviesService, private route: ActivatedRoute) { }

  ngOnInit(): void {

    this.movie = this.route.snapshot.paramMap.get("movie")!;
    console.log('similar_movie: '+this.movie);
    this._getAllTeams();
    //getPopularMovies(): Promise<Movie>

  }

  _getAllTeams():void{
    this.moviesService.getSimilarMovies(this.movie)
      .then((movies)=>this._fillTeams(movies))
      .catch((error)=>this._handleError(error))
  }

  
  _fillTeams(movies: Movie[]){
    this.movies = movies;
    console.log("_fillTeams==movies: "+ JSON.stringify(this.movies));
    return this.movies;
  }

  
  _handleError(error: any){
    console.log(error);
    alert(error.message)

  }
}
