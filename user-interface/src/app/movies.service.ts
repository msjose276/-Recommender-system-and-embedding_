import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormGroup } from '@angular/forms';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class MoviesService {

  private baseURL: String = "http://localhost:3000/api/";

  constructor(private http: HttpClient) { }

  public getPopularMovies(): Promise<Movie[]> {
    const url = this.baseURL+"trendingMovies";// + "users/login";//+jobID;

    return this.http.get<Movie[]>(url).toPromise()
      .then(response => response as Movie[])
      .catch(this.handleError);
  }

  public getSimilarMovies(movie: String): Promise<Movie[]> {
    const url = this.baseURL+"similarMovies";// + "users/login";//+jobID;

    console.log("title="+movie)
    return this.http.post<Movie[]>(url,{"title":movie}).toPromise()
      .then(response => response as Movie[])
      .catch(this.handleError);
  }

  private handleError(error: any):Promise<any> {
    return Promise.reject(error.message || error);
  }

}


export class Movie{

  id!:String;
  original_language!:String;
  title!:String;
  
  #success!:String;
  #token!:String;
  /*
  constructor(success: String, token: String){
    this.#success =success;
    this.#token = token;
  }*/
  get token(){return this.#token}
  get success(){return this.#success}
}