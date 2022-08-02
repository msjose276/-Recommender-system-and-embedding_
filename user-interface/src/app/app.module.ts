import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { AppComponent } from './app.component';
import { PopularMoviesComponent } from './popular-movies/popular-movies.component';
import { SimilarMoviesComponent } from './similar-movies/similar-movies.component';
import { NavegationComponent } from './navegation/navegation.component';

@NgModule({
  declarations: [
    AppComponent,
    PopularMoviesComponent,
    SimilarMoviesComponent,
    NavegationComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    ReactiveFormsModule,
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
