import './App.css';
import React from 'react';
import {Link} from 'react-router-dom'

// link will allow us to click and switch between tabs

function Navigation() {

    const navStyle={
      color: 'white'   
    };

  return (
   <nav>
       <h3>Evaluating Classifiers</h3>

       <ol className="Nav-links">
    
           <Link style={navStyle} to='/'>
            <li>Research</li>
           </Link>

           <Link style={navStyle} to='/predict'>
            <li>Predict</li>
           </Link>
           
       </ol>

   </nav>
  );
}

export default Navigation;
