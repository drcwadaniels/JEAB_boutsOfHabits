

// stan code for biexponential

data {

 int<lower=1> Nr; // number of data points
 int<lower=1> Ns; // number of subjects
 vector<lower=0>[Nr] y; //irts
 int s[Nr]; //subjects
 

}

parameters {

//parameters
real beta_short_mu;
real delta_long_mu; 
real L_mu;

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

  beta_short = exp(beta_short_temp+beta_short_mu); 
  delta_long = exp(delta_long_temp+delta_long_mu);
  L = exp(L_temp+L_mu);
  
  beta_long = beta_short + delta_long; 
  Q = exp(L)./(1+exp(L));

}

model {

//priors
//mean
target+= normal_lpdf(beta_short_mu|0.3,0.2);
target+= normal_lpdf(delta_long_mu|2,1.5);
target+= normal_lpdf(L_mu|5,4);


//subject specific shift from mean
target+= normal_lpdf(short_rme|0,0.2);
target+= normal_lpdf(long_rme|0,1.5);
target+= normal_lpdf(L_rme|0,4); 




//likelihood

for (i in 1:Nr)
{
  target+= log_sum_exp(log1m(Q[i])+exponential_lpdf(y[i]|beta_short[i]),log(Q[i])+exponential_lpdf(y[i]|beta_long[i])); //biexponential likelihood
}


}


