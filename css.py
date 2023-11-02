import streamlit as st


def load_css_styles():
  
  st.markdown("""
  <style>
    @import url(https://fonts.googleapis.com/css?family=Open+Sans:400,700,400italic,700bold);

    /* Sidebar top margin */
    .st-emotion-cache-ysnqb2.ea3mdgi4 {
      margin-top: -30px;
      margin-bottom: -30px;
      /* margin-top: -75px;
      margin-bottom: -90px; */
    } 

    /* Front page text */
    div.frontpage {
      text-align: center;
      width: 80%;
      margin: auto;
    }

    /* Sidebar padding 
    .st-emotion-cache-10oheav {
        padding: 2rem 1rem;
    } 
    */


    /* Main panel padding */
    .st-emotion-cache-z5fcl4 {
      padding-bottom: 2rem;
    }

    .list_group-item-more {
      background-color: #494950;
      border: #494950;
    }
    .list_group-item-more:hover {
      background-color: #F63366;
      color: #FFFFFF;
    }
    .list_group-item-more:active {
      background-color: #F63366;
      border: #FFFFFF;
    }


/* Sidebar */


  .sidebar {
  margin: 0;
  width: 100%;
  padding: 0px;
  /* background-color: #494950; */
  position: relative;
  height: 100%;
  overflow: auto;
  text-indent: 10px;
}

.sidebar a:link {
  display: block;
  color: white;
  padding: 9px;
  text-decoration: none;
  list-style-type: circle;
}
 
.sidebar a:active {
  background-color: #F63366;
  color: white;
}

.sidebar a:hover:not(.active) {
  /* background-color: #555; */
  background-color: #F63366;
  color: white;
}

a.nav {
  font-size: 97%;
  color: white;
}

h3.nav2 {
  font-size: 97%;
}

  
  /* Down Arrow */
  /* https://codepen.io/pjwiebe/pen/VmmxpM */
  
  .arrow {
    margin-top: 0px;
    margin-bottom: 30px;
    box-sizing: border-box;
    height: 3vw;
    width: 3vw;
    border-style: solid;
    border-color: white;
    border-width: 0px 2px 2px 0px;
    transform: rotate(45deg);
    transition: border-width 150ms ease-in-out;
  }
  
  .arrow:hover {
    border-color: #FF9F36;
    border-bottom-width: 5px;
    border-right-width: 5px;
  }

  .up-arrow {
    margin-top: 60px;
    box-sizing: border-box;
    height: 3vw;
    width: 3vw;
    border-style: solid;
    border-color: white;
    border-width: 0px 2px 2px 0px;
    transform: rotate(225deg);
    transition: border-width 150ms ease-in-out;
  }
  
  .up-arrow:hover {
    border-color: #FF9F36;
    border-bottom-width: 5px;
    border-right-width: 5px;
  }
  
  .container {
    display: flex;
    align-items: center;
    justify-content: center;
    scroll-snap-type: y mandatory;
  }
  
  #full {
    margin-top:60px;
    height: 90vh;
    background-color: #29b5e8;
    scroll-snap-type: y mandatory;
  }
  
  .child {
    scroll-snap-align: start;
  }
  
  /* Scroll Snapping */
  /* https://codepen.io/tutsplus/pen/qpJYaK */
  
  .scroll-container,
  .scroll-area {
    /* max-width: 850px; */
    max-width: 100%;
    height: 70vh;
    font-size: 60px;
  }
  
  .scroll-container {
    overflow: auto;
    scroll-snap-type: y mandatory;
  }
  
  .scroll-area {
    scroll-snap-align: start;
  }
  
  .scroll-container,
  .scroll-area {
    margin: 0 auto;
  }
  
  .scroll-area {
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
  }
  
  .scroll-area:nth-of-type(1) {
    background: #49b293;
  }
  
  .scroll-area:nth-of-type(2) {
    background: #c94e4b;
  }
  
  .scroll-area:nth-of-type(3) {
    background: #4cc1be;
  }
  
  .scroll-area:nth-of-type(4) {
    background: #8360A6;
  }
  
  #first {
    margin-top:0px;
  }

  #second {
    margin-top:0px;
  }

  #third {
    margin-top:0px;
  }

  #fourth {
    margin-top:0px;
  }

  #fifth {
    margin-top:0px;
  }

  #sixth {
    margin-top:0px;
  }

  #seventh {
    margin-top:0px;
  }

  #eighth {
    margin-top:0px;
  }

  #ninth {
    margin-top:0px;
  }

  #tenth {
    margin-top:0px;
  }

  #eleventh {
    margin-top:0px;
  }

  #twelfth {
    margin-top:0px;
  }
  
  #top {
    margin-top:0px;
  }


/*
  [data-testid="block-container"] {
     margin-top:0px;
  }
*/



  .avatar {
    vertical-align: middle;
    width: 20px;
    height: 20px;
    border-radius: 50%;
  }

  a:link, a:visited {
    /* color: #29b5e8; */
    text-decoration: none;
  }

  a:hover, a:active {
    color: #F63366;
    text-decoration: none;
  }

  # img:hover {transform: scale(1.05);}
  .img_hover {transform: scale(1.05);}

  
  </style>
  """, unsafe_allow_html=True)


  
 # body{
 #   text-align:center
 # }

def load_footer():
  # Social icons https://codepen.io/kpdushanmaduka/pen/jadeoO
  css_styles = '''

  
  i {
    color: #FFFFFF;
    letter-spacing: 9px;
    padding: 0.5cm 0cm 0.5cm 0cm;
    font-size: 20px;
    transition: all .2s ease-in-out;
  }
  
  i:hover {transform: scale(1.5);}
  
  .fa-github:hover {color: #B3B6B7;}
  .fa-instagram:hover {color: #c32aa3;}
  .fa-twitter:hover {color: #1da1f2;}
  .fa-linkedin-in:hover {color: #0a66c2;}
  .fa-youtube:hover {color: #ff0000;}
  '''
  st.markdown(f'<style>{css_styles}</style>', unsafe_allow_html=True)
      
  # Font awesome icon
  fa_css = '''
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
  
  <p align="center">
      <span style="color: #FFBD45;">Follow:</span>&nbsp;
      <a href="https://github.com/streamlit"><i class="fab fa-github"></i></a>
      <a href="https://twitter.com/streamlit/"><i class="fa-brands fa-x-twitter"></i></a>
      <a href="https://www.linkedin.com/company/streamlit/"><i class="fab fa-linkedin-in"></i></a>
      <a href="https://www.youtube.com/@streamlitofficial"><i class="fab fa-youtube"></i></a>
      <a href="https://www.instagram.com/streamlit.io/"><i class="fab fa-instagram"></i></a>
  </p>
  '''      
  # fab fa-twitter
  
  st.write(fa_css, unsafe_allow_html=True)
