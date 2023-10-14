

// stan code for biexponential

data {

 int<lower=1> Nr; // number of data points
 int<lower=1> Ns; // number of subjects
 int<lower=1> Np; // number of predictors
 vector<lower=0>[Nr] y; //irt stream
 vector<lower=0>[Nr] cens_threshold; //threshold for censored data
 int s[Nr]; //subject stream
 int C[Nr]; //censor stream
 int<lower=1,upper=Np+1> P[Nr]; //predictor stream


}

parameters {

//parameters
real beta_short_mu;
real delta_long_mu; 
real L_mu;

vector[Np] betas_beta_short;
vector[Np] betas_delta_long;
vector[Np] betas_L; 

vector[Ns] short_rme;
vector[Ns] long_rme;
vector[Ns] L_rme;

vector[Ns] sess_short_rme;
vector[Ns] sess_long_rme;
vector[Ns] sess_L_rme;


}

transformed parameters {
vector[Nr] beta_short_temp;
vector[Nr] delta_long_temp;
vector[Nr] L_temp;

vector<lower=0>[Nr] beta_short;
vector<lower=0>[Nr] delta_long; 
vector<lower=0>[Nr] beta_long;
vector<lower=0>[Nr] L;
vector<lower=0,upper=1>[Nr] Q; 



for (i in 1:Nr)
{

  
  if (P[i]==1)
  {
  //condition = 0, session = 0, condition*session = 0
    beta_short_temp[i] = (beta_short_mu + short_rme[s[i]]);
    delta_long_temp[i] = (delta_long_mu + long_rme[s[i]]);
    L_temp[i] = (L_mu + L_rme[s[i]]);    
  }
  else if (P[i]==2)
  {
  //condition = 1, session = 0, condition*session = 0
    beta_short_temp[i] = (beta_short_mu + short_rme[s[i]]) + 1*betas_beta_short[P[i]-1];
    delta_long_temp[i] = (delta_long_mu + long_rme[s[i]]) + 1*betas_delta_long[P[i]-1];
    L_temp[i] = (L_mu + L_rme[s[i]]) +  1*betas_L[P[i]-1];   
  }
  else if (P[i] == 3) 
  {
  //condition = 1, session = 1, condition*session = 0
    beta_short_temp[i] = (beta_short_mu + short_rme[s[i]]) + 1*(betas_beta_short[P[i]-1]+sess_short_rme[s[i]]);
    delta_long_temp[i] = (delta_long_mu + long_rme[s[i]]) + 1*(betas_delta_long[P[i]-1]+sess_long_rme[s[i]]);
    L_temp[i] = (L_mu + L_rme[s[i]]) + + 1*(betas_L[P[i]-1]+sess_L_rme[s[i]]); 
  }
  else if (P[i]==4)
  {
  //condition = 1, session = 1, condition*session = 1
    beta_short_temp[i] = (beta_short_mu + short_rme[s[i]]) + 1*betas_beta_short[1] + 1*(betas_beta_short[2]+sess_short_rme[s[i]]) +  1*betas_beta_short[P[i]-1];
    delta_long_temp[i] = (delta_long_mu + long_rme[s[i]]) + 1*betas_delta_long[1] + 1*(betas_delta_long[2]+sess_long_rme[s[i]]) +  1*betas_beta_short[P[i]-1];
    L_temp[i] = (L_mu + L_rme[s[i]]) +  1*betas_L[1] + 1*(betas_L[2]+sess_L_rme[s[i]]) +  1*betas_beta_short[P[i]-1];   

  }

  
  
}

  beta_short = exp(beta_short_temp); 
  delta_long = exp(delta_long_temp);
  L = exp(L_temp);
  
  beta_long = beta_short + delta_long; 
  Q = L./(1+L);

}

model {

//priors
//mean
target+= normal_lpdf(beta_short_mu|-1,3);
target+= normal_lpdf(delta_long_mu|2.5,4);
target+= normal_lpdf(L_mu|0,3);

//betas fixed effects
target+= normal_lpdf(betas_beta_short|0,3);
target+= normal_lpdf(betas_delta_long|0,3);
target+= normal_lpdf(betas_L|0,3);


//subject specific shift from mean
target+= normal_lpdf(short_rme|0,2);
target+= normal_lpdf(long_rme|0,2);
target+= normal_lpdf(L_rme|0,2); 

//subject specific shift from mean session slope
target+= normal_lpdf(sess_short_rme|0,2);
target+= normal_lpdf(sess_long_rme|0,2);
target+= normal_lpdf(sess_L_rme|0,2); 


//likelihood

for (i in 1:Nr)
{
  if (C[i]==0)
  {
    //biexponential for uncensored data
    target+= log_sum_exp(log1m(Q[i])+exponential_lpdf(y[i]|1/beta_short[i]),log(Q[i])+exponential_lpdf(y[i]|1/beta_long[i])); 
  }
  else 
  {
    //biexponential for censored data
    target+= log_sum_exp(log1m(Q[i])+exponential_lccdf(cens_threshold[i]|1/beta_short[i]),log(Q[i])+exponential_lccdf(cens_threshold[i]|1/beta_long[i]));
  }

}


}


