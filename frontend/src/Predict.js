import { useEffect, useState } from 'react';
import './App.css';
import React from 'react';


function Predict() {

    const [status,setStatus] = useState('');

    const [accuracy,setAccuracy] = useState('');

    const [normalize,setNormalize] = useState('');

    function updateNormalize(){
      var selectBox = document.getElementById("select_normalize");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setNormalize(selectedValue);
    }

    const [feature_technique,setFeatureTechnique] = useState('')

    function updateFeatureTechnique(){
      var selectBox = document.getElementById("select_feature_technique");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setFeatureTechnique(selectedValue);
    }

    const [kvalue,setKValue] = useState('')

    function updateKValue(){
      var selectBox = document.getElementById("select_k");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setKValue(selectedValue);
    }

    const [classifier,setClassifier] = useState('');

    function updateClassifier(){
      var selectBox = document.getElementById("select_classifier");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setClassifier(selectedValue);
    }

    const [time,setTime] = useState('')

    const timeHandler = (e) => {
      setTime(e.target.value);
    }
  
    const [age,setAge] = useState('')

    const ageHandler = (e) => {
      setAge(e.target.value);
    }

    const [ejection,setEjection] = useState('')

    const ejectionHandler = (e) => {
      setEjection(e.target.value);
    }

    const [sodium,setSodium] = useState('')

    const sodiumHandler = (e) => {
      setSodium(e.target.value);
    }

    const [creatinine,setCreatinine] = useState('')

    const creatinineHandler = (e) => {
      setCreatinine(e.target.value);
    }

    const [pletelets,setPletelets] = useState('')

    const pleteletsHandler = (e) => {
      setPletelets(e.target.value);
    }

    const [cpk,setCPK] = useState('')

    const cpkHandler = (e) => {
      setCPK(e.target.value);
    }

    const [gender,setGender] = useState('')

    function updateGender(){
      var selectBox = document.getElementById("select_gender");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setGender(selectedValue);
    }

    const [smoking,setSmoking] = useState('')

    function updateSmoking(){
      var selectBox = document.getElementById("select_smoking");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setSmoking(selectedValue);
    }

    const [diabetes,setDiabetes] = useState('')

    function updateDiabetes(){
      var selectBox = document.getElementById("select_diabetes");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setDiabetes(selectedValue);
    }

    const [anamia,setAnamia] = useState('')

    function updateAnamia(){
      var selectBox = document.getElementById("select_anamia");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setAnamia(selectedValue);
    }

    const [bp,setBP] = useState('')

    function updateBP(){
      var selectBox = document.getElementById("select_bp");
      var selectedValue = selectBox.options[selectBox.selectedIndex].value;
      setBP(selectedValue);
    }

    function displayValues(){
      console.log(time,age,sodium,creatinine,ejection,cpk,pletelets,smoking,gender,anamia,diabetes,bp);
    }

  return (
    <div className="Predict_Page">
        <h1 className="Predict-Heading">Make A Prediction</h1>

        <div className="container">

        <div className="box">
          <label  >Normalize:  </label>   
          <select id="select_normalize" onChange={updateNormalize}>
            <option   value="true" >True</option>
            <option   value="false" >False</option>
          </select>
        </div>
        
        <div className="box">
          <label  >Feature Technique:  </label>   
          <select id="select_feature_technique" onChange={updateFeatureTechnique}>
            <option  value="filter" >Filter</option>
            <option  value="wrapper"  >Wrapper</option>
            <option  value="hybrid" >Hybrid</option>
            <option  value="pca" >PCA</option>
          </select>
      </div>

        <div className="box">
          <label  >Classifier:  </label>   
          <select id="select_classifier" onChange={updateClassifier}>
            <option  value="Naive Bayes" >Naive Bayes</option>
            <option  value= "K-Nearest Neighbours">K-Nearest Neighbours</option>
          </select>
      </div>

            { (classifier==="K-Nearest Neighbours")?
            <div className="box">
              <label  >Select Value of K: </label>   
              <select id="select_k" onChange={updateKValue}>
              <option  value="1"  >1</option>
              <option  value="3"  >3</option>
              <option  value="5"  >5</option>
              <option  value="7"  >7</option>
              <option  value="9"  >9</option>
              <option  value="best">Best K (Recommended)</option>
            </select>
            </div>
            :false
          }
      </div>  
        
      <h1 className="Predict-Heading">Enter Test Case Values</h1>

      <div className="container">

        <input style={{"width":"150px"}} className="input-field" value={time} onChange={timeHandler} type="text" placeholder="Enter Time..."   />
        <input style={{"width":"150px"}} className="input-field" value={age} onChange={ageHandler} type="text" placeholder="Enter Age..."   />
        <input style={{"width":"150px"}} className="input-field" value={sodium} onChange={sodiumHandler} type="text" placeholder="Enter Sodium ..."   />
        <input style={{"width":"150px"}} className="input-field" value={creatinine} onChange={creatinineHandler} type="text" placeholder="Enter Creatinine..."   />
        <input style={{"width":"150px"}} className="input-field" value={ejection} onChange={ejectionHandler} type="text" placeholder="Enter Ejection..."   />
        <input style={{"width":"150px"}} className="input-field" value={cpk} onChange={cpkHandler} type="text" placeholder="Enter CPK..."   />

      </div>

      <br/>      <br/>      <br/>

      <div className="container">

        <input style={{"width":"150px"}} className="input-field" value={pletelets} onChange={pleteletsHandler} type="text" placeholder="Enter Pletelets..."   />

      <div className="box">
        <label  >Smoking:  </label>   
          <select id="select_smoking" onChange={updateSmoking}>
            <option  value="1"  >True</option>
            <option  value="0"  >False</option>
          </select>
      </div>

      <div className="box">
          <label  >Gender:  </label>   
          <select id="select_gender" onChange={updateGender}>
            <option  value="1" >Male</option>
            <option  value="0" >Female</option>
          </select>
      </div>

      <div className="box">
          <label  >Anamia:  </label>   
          <select id="select_anamia" onChange={updateAnamia}>
            <option  value="1"  >Yes</option>
            <option  value="0"  >No</option>
          </select>
      </div>

      <div className="box">
          <label  >Diabetes:  </label>   
          <select id="select_diabetes" onChange={updateDiabetes}>
            <option  value="1"  >Yes</option>
            <option  value="0"  >No</option>
          </select>
      </div>

      <div className="box">
          <label  >BP:  </label>   
          <select id="select_bp" onChange={updateBP}>
            <option  value="1"  >Yes</option>
            <option  value="0"  >No</option>
          </select>   
      </div>
      </div>

      <br/>      <br/>      <br/>

      <div className="container">
        <button className="predict_button" type="button" onClick={

                    async()=>{

                      var selectBox = document.getElementById("select_normalize");
                      var normalize_var = selectBox.options[selectBox.selectedIndex].value;

                      var selectBox = document.getElementById("select_feature_technique");
                      var feature_technique_var = selectBox.options[selectBox.selectedIndex].value;

                      var selectBox = document.getElementById("select_classifier");
                      var classifier_var = selectBox.options[selectBox.selectedIndex].value;

                      var kvalue_var='';
                      if (classifier_var==="K-Nearest Neighbours"){
                        var selectBox = document.getElementById("select_k");
                        kvalue_var = selectBox.options[selectBox.selectedIndex].value;
                      }else{
                        kvalue_var='NONE';
                      }

                      var selectBox = document.getElementById("select_smoking");
                      var smoking_var = selectBox.options[selectBox.selectedIndex].value;

                      var selectBox = document.getElementById("select_gender");
                      var gender_var = selectBox.options[selectBox.selectedIndex].value;

                      var selectBox = document.getElementById("select_anamia");
                      var anamia_var = selectBox.options[selectBox.selectedIndex].value;

                      var selectBox = document.getElementById("select_diabetes");
                      var diabetes_var = selectBox.options[selectBox.selectedIndex].value;

                      var selectBox = document.getElementById("select_bp");
                      var bp_var = selectBox.options[selectBox.selectedIndex].value;


                        const Prediction_Details={
                            normalize_var,
                            feature_technique_var,
                            classifier_var,
                            kvalue_var,
                            time,
                            age,
                            sodium,
                            creatinine,
                            ejection,
                            cpk,
                            pletelets,
                            smoking_var,
                            gender_var,
                            anamia_var,
                            diabetes_var,
                            bp_var
                        };
                        const response=await fetch('/make_prediction',{
                            method: "POST",
                            headers: {
                                "Content-Type":"application/json"
                            },
                            body: JSON.stringify(Prediction_Details)
                        }).then(response=>response.json().then(data=>{

                            setAccuracy(data['accuracy']);
                            if(data['predicted_value']){
                              setStatus('Dead');
                            }else{
                              setStatus('Alive');
                            }                                                      
                        }));

                    }

        } 
        >Predict</button>
      </div>
      
      {(status!=='')?
        <div>
          <h3 className="Predict-Heading">Patient Status: {status}</h3>
          <h3>Calculated with accuracy: {accuracy*100} %</h3>
        </div>
        :false
      } 
    </div>  
  );
}

export default Predict;
