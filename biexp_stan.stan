

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
  beta_short_temp[i] = short_rme[s[i]];
  delta_long_temp[i] = long_rme[s[i]];
  L_temp[i] = L_rme[s[i]]; 
  
  
}

  beta_short = exp(beta_short_mu+(P*betas_beta_short)+beta_short_temp); 
  delta_long = exp(delta_long_mu+(P*betas_delta_long)+delta_long_temp);
  L = exp(L_mu+(P*betas_L)+L_temp);
  
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


//subject specific shift from mean
target+= normal_lpdf(short_rme|0,2);
target+= normal_lpdf(long_rme|0,2);
target+= normal_lpdf(L_rme|0,2); 


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


