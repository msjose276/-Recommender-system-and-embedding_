

const express = require('express');
const Router = express.Router();

const similarMovies = require('../controllers/similarMovies');
const trending = require('../controllers/trending');
const recommendations = require('../controllers/recommendations');

/*
Router.route('/')
    .get(trending.trending)
    .post(similarMovies.similarMovies)
    */
Router.route('/trendingMovies')
    .get(trending.trending)
Router.route('/similarMovies')
    .post(similarMovies.similarMovies)

Router.route('/factorizationSVDRecommendationMatrix')
    .post(recommendations.factorizationSVDRecommendationMatrix)
   

Router.route('/NeuralCollaborativerecommendation')
    .post(recommendations.NeuralCollaborativerecommendation)

//Router.route('/recommendationBasedOnUserAndItem')
  //  .post(recommendations.recommendationBasedOnUserAndItem)

Router.route('/binaryMatrixRecommendationBasedOnUser')
    .post(recommendations.binaryMatrixRecommendationBasedOnUser)
     
    
    
module.exports = Router;