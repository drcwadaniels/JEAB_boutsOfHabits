

// stan code for biexponential

data {

 int<lower=1> Nr; // number of data points
 int<lower=1> Ns; // number of subjects
 int<lower=1> Np; // number of predictors
 vector<lower=0>[Nr] y; //irt stream
 vector<lower=0>[Nr] cens_threshold; //threshold for censored data
 int s[Nr]; //subject stream
 int C[Nr]; //censor stream
 matrix[Nr,Np] P; //predictor matrix


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

//random slope for effect of session
vector[Ns] short_sess_rme;
vector[Ns] long_sess_rme;
vector[Ns] L_sess_rme;




}

transformed parameters {
vector[Nr] beta_short_temp;
vector[Nr] delta_long_temp;
vector[Nr] L_temp;

vector[Nr] short_predictors;
vector[Nr] long_predictors;
vector[Nr] L_predictors;

vector<lower=0>[Nr] beta_short;
vector<lower=0>[Nr] delta_long; 
vector<lower=0>[Nr] beta_long;
vector<lower=0>[Nr] L;
vector<lower=0,upper=1>[Nr] Q; 



for (i in 1:Nr)
{
  beta_short_temp[i] = short_rme[s[i]];
  delta_long_temp[i] = long_rme[s[i]];
  L_temp[i] = L_rme[s[i]]; 
  
  if (P[i,2]==1)
  {
    short_predictors[i] = betas_beta_short[2] +  short_sess_rme[s[i]];
    long_predictors[i] = betas_delta_long[2] + long_sess_rme[s[i]];
    L_predictors[i] = betas_L[2] + L_sess_rme[s[i]];
  }
  else 
  {
    if (P[i,3]==1)
    {
      short_predictors[i] = betas_beta_short[3];
      long_predictors[i] = betas_beta_short[3];
      L_predictors[i] = betas_beta_short[3];
    }
  else 
    {
      short_predictors[i] = betas_beta_short[1];
      long_predictors[i] = betas_beta_short[1];
      L_predictors[i] = betas_beta_short[1];  
    }

  }

}

  beta_short = exp((beta_short_mu+beta_short_temp)+(P*short_predictors)); 
  delta_long = exp((delta_long_mu+delta_long_temp)+(P*long_predictors));
  L = exp((L_mu+L_temp)+(P*L_predictors));
  
  beta_long = beta_short + delta_long; 
  Q = L./(1+L);

}

model {

//priors
//mean
target+= normal_lpdf(beta_short_mu|-1,3);
target+= normal_lpdf(delta_long_mu|2.5,4);
target+= normal_lpdf(L_mu|0,3);

//betas
target+= normal_lpdf(betas_beta_short|0,3);
target+= normal_lpdf(betas_delta_long|0,3);
target+= normal_lpdf(betas_L|0,3);


//subject specific shift from intercept
target+= normal_lpdf(short_rme|0,2);
target+= normal_lpdf(long_rme|0,2);
target+= normal_lpdf(L_rme|0,2); 

//subject specific shift from session slope
target+= normal_lpdf(short_sess_rme|0,1);
target+= normal_lpdf(long_sess_rme|0,1);
target+= normal_lpdf(L_sess_rme|0,1);


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


