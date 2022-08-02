import { NgModule } from "@angular/core";
import { RouterModule, Routes } from "@angular/router";
import { PopularMoviesComponent } from "./popular-movies/popular-movies.component";
import { SimilarMoviesComponent } from "./similar-movies/similar-movies.component";

const routes: Routes = [
    {path:'', redirectTo:'/popular-movies',pathMatch:'full'},
    //{path:'home-page', redirectTo:'/team-page',pathMatch:'full'},
    //{path:'update-team/:teamID',component:CreateTeamComponent},

    {path:'popular-movies',component:PopularMoviesComponent},
    {path:'similar-movies',component:SimilarMoviesComponent},
    {path:'similar-movies/:movie',component:SimilarMoviesComponent}

];

@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
})

export class AppRoutingModule{}