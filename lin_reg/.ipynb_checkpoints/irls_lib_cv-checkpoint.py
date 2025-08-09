import numpy as np
import statsmodels.api as sm
from scipy.sparse.linalg import eigsh
import time
from sklearn.base import BaseEstimator

def tukeyBisquare( X, y, w_star ):
    start_time=time.time()
    n, d = X.shape
    mod_wls = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    res_wls = mod_wls.fit()
    w = res_wls.params.reshape(d,1)
    return [ np.linalg.norm( w - w_star ), time.time() - start_time ]

def Filter(tau,res,c):
    sigma=np.mean(np.power(res,2))
    if sum(tau) < c*sigma:
        return None
    else:
        tau_threshold=np.max(tau)*np.random.uniform()
        return np.where(tau < tau_threshold)[0]

#Robust Estimation via Robust Gradient Estimation

def findInterval(X,k):
    if k>1:
        x=np.sort(X,axis=None)
        y=[x[i+k-1] -x[i] for i in range(0,len(x)-k+1)]
        j=np.argmin(y)
        return x[j:j+k].reshape(-1,1)
    else:
        return np.array(X[0]).reshape(-1,1)

def findBall(X,a,k):
    if k>0:
        t=np.linalg.norm(X-a.reshape(-1),axis=1)
        return X[np.argsort(t)[:k],:]
    else:
        return a.reshape(-1,1)
    
def OutlierGradTrunc(grad, alpha, delta):
    n,d=grad.shape
        
    if d==1:
        #k=max(int((1-alpha -np.sqrt(np.log(n/delta)/n))*(1-alpha)*n),1)
        k = max(int((1-alpha)*(1-alpha)*n), 1)
        #print( "k = ", k )
        return findInterval(grad,k)
    else:
        a=np.array([RobustGrad(grad[:,i].reshape(-1,1),alpha,delta/d) for i in range(d)])
        #k=max(int((1-alpha -np.sqrt((d/n)*np.log(n/(delta*d))))*(1-alpha)*n),1)
        k = max(int((1-alpha)*(1-alpha)*n),1)
        #print( "k = ", k )
        return findBall(grad, a, k)
		
def RobustGrad(grad, alpha, delta):
    n,d=grad.shape
    #print(grad.shape)
    grad=OutlierGradTrunc(grad, alpha, delta)
    if d==1:
        return np.mean(grad, axis=0).reshape(-1,1)
    else:
        _,U=np.linalg.eigh(np.matmul(grad.transpose(),grad))
        #split = int(d/2)
        split =d//2
        V=U[:, split:] #top-half eigen vectors      d*(d-split)
        W=U[:,:split]  #bottom-half eigen vectors   d*split
        mu_v=RobustGrad(np.matmul(grad,V), alpha, delta)         #(d-split)*1
        mu_w=np.mean(np.matmul(grad,W), axis=0).reshape(-1,1)    #split*1
#        print( mu_v.shape, mu_w.shape )
#        print( mu_v, mu_w, split, d )
#        print( V.shape, W.shape )
        #return V.dot( mu_v ) + W.dot( mu_w )
        return np.matmul(V,mu_v)+np.matmul(W,mu_w) #d*1

def HT(a,k):
    t=np.zeros(a.shape)
    if k==0:
        return t
    else:
        ind=np.argpartition(abs(a),-k, axis=None)[-k:]    
        t[ind,:]=a[ind,:]
    return t


class STIR(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, eta = 1.01, alpha = 0.0, M_init = 10.0, w_init = None, w_star = None ):
		self.eta = eta
		self.alpha = alpha
		self.M_init = M_init
		self.w_init = w_init
		self.w_star = w_star
	
	def fit( self, X, y, max_iter = 40, max_iter_w = 1 ):
		start_time=time.time()
		n, d = X.shape
		M = self.M_init
		self.w = self.w_init
		
		self.l2=[]
		self.clock=[]
		itr=0
		
		while itr < max_iter:        
			iter_w = 0
			while iter_w < max_iter_w:
				s = abs( np.dot( X, self.w ) - y )
				np.clip( s, 1 / M, None, out = s )        
				s = 1/s
				
				mod_wls = sm.WLS( y, X, weights = s )
				res_wls = mod_wls.fit()
				self.w = res_wls.params.reshape( d, 1 )
				
				iter_w += 1     
				self.l2.append( np.linalg.norm( self.w - self.w_star ) )
				self.clock.append( time.time() - start_time )
							
				if iter_w >=max_iter_w:
					break
			itr += iter_w
			M *= self.eta
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )
    
    
class SVAM(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, eta = 1.01, alpha = 0.0, beta_init = 10.0, w_init = None, w_star = None ):
		self.eta = eta
		self.alpha = alpha
		self.beta_init = beta_init
		self.w_init = w_init
		self.w_star = w_star
	
	
	def fit( self, X, y, max_iter = 40, max_iter_w = 1 ):
		start_time=time.time()
		n, d = X.shape
		beta = self.beta_init
		self.w = self.w_init
		self.l2=[]
		self.clock=[]
		itr=0
		
		while itr < max_iter:        
			iter_w = 0
			while iter_w < max_iter_w:
				s=np.power(beta/2*np.pi,0.5)*np.exp(-beta/2 *np.power(np.dot(X,self.w)-y,2)) 
				mod_wls = sm.WLS( y, X, weights = s )
				res_wls = mod_wls.fit()
				self.w = res_wls.params.reshape( d, 1 )
				iter_w += 1     
				self.l2.append( np.linalg.norm( self.w - self.w_star ) )
				self.clock.append( time.time() - start_time )
							
				if iter_w >=max_iter_w:
					break
			itr += iter_w
			beta*= self.eta
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )    

class TORRENT(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, alpha = 0.0, w_init = None, w_star = None ):
		self.alpha = alpha
		self.w_init = w_init
		self.w_star = w_star
	
	
	def fit( self, X, y, max_iter = 6 ):
		start_time = time.time()
		n, d = X.shape    
		n_clean = int( ( 1 - self.alpha ) * n) # number of points we think are clean
		cleanIdx = np.arange(n)
		self.l2=[]
		self.clock=[]
		for t in range(max_iter):
			mod_ols = sm.OLS( y[cleanIdx], X[cleanIdx,:] )
			res_ols = mod_ols.fit()
			self.w = res_ols.params.reshape(d,1)
			
			res = abs( np.dot( X, self.w ) - y )
			cleanIdx = sorted( range( len( res ) ), key = lambda k: res[k] )[ 0: n_clean ]
			self.l2.append( np.linalg.norm( self.w - self.w_star ) )
			self.clock.append(time.time()-start_time)
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )
		
class SEVER(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, alpha = 0.0, w_init = None, w_star = None ):
		self.alpha = alpha
		self.w_init = w_init
		self.w_star = w_star
	
# 	def get_params( self, deep = True ):
# 		return { "alpha": self.alpha, "w_init": self.w_init, "w_star": self.w_star }
	
# 	def set_params( self, **params ):
# 		for key, value in params.items():
# 			setattr( self, key, value )
	
	def fit( self, X, y, max_iter = 30 ):
		start_time=time.time()
		n, d = X.shape    
		per_iter = int( np.ceil( self.alpha * n / max_iter ) ) +  d # number of points thrown out after each iteration
		cleanE = np.arange(n)
		self.l2=[]
		self.clock=[]
		for i in range( max_iter ):
			X = X[cleanE,:]
			y = y[cleanE,:]
			mod_ols = sm.OLS(y, X)
			res_ols = mod_ols.fit()
			self.w = res_ols.params.reshape( d, 1 )
			self.l2.append(np.linalg.norm( self.w - self.w_star ) )
			res = np.matmul( X , self.w ) - y
			G = X * res - np.matmul( X.transpose(), res ).reshape(-1) / cleanE.shape[0]
			_,v = eigsh( np.matmul( G.transpose(), G ), k = 1 )
			tau = np.power( np.matmul( G, v ), 2 )
			cleanE = Filter( tau, res, 200 )
			self.clock.append(time.time()-start_time)
			
			if cleanE is None:
				break
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )

class SEVER_M(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, alpha = 0.0, w_init = None, w_star = None ):
		self.alpha = alpha
		self.w_init = w_init
		self.w_star = w_star
	
# 	def get_params( self, deep = True ):
# 		return { "alpha": self.alpha, "w_init": self.w_init, "w_star": self.w_star }
	
# 	def set_params( self, **params ):
# 		for key, value in params.items():
# 			setattr( self, key, value )
	
	def fit( self, X, y, max_iter = 30 ):
		start_time=time.time()
		n, d = X.shape    
		per_iter = int( np.ceil( self.alpha * n / max_iter ) ) +  d # number of points thrown out after each iteration
		cleanE = np.arange(n)
		self.l2=[]
		self.clock=[]
		for i in range( max_iter ):
			X = X[cleanE,:]
			y = y[cleanE,:]
			mod_ols = sm.OLS(y, X)
			res_ols = mod_ols.fit()
			self.w = res_ols.params.reshape(d,1)
			self.l2.append( np.linalg.norm( self.w- self.w_star ) )
			res = np.matmul( X, self.w ) - y
			G = X * res - np.matmul( X.transpose(), res ).reshape(-1) / cleanE.shape[0]
			_,v = eigsh( np.matmul( G.transpose(), G ), k = 1 )
			tau = np.power( np.matmul( G, v ), 2 )
			cleanE = tau.argsort( axis = None )[ : len( tau ) - per_iter ]			
			self.clock.append(time.time()-start_time)
			
			if cleanE is None:
				break
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )

class RGD(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, alpha = 0.0, delta = 0.1, eta = 0.1, w_init = None, w_star = None ):
		self.alpha = alpha
		self.delta = delta
		self.eta = eta
		self.w_init = w_init
		self.w_star = w_star
	
# 	def get_params( self, deep = True ):
# 		return { "alpha": self.alpha, "w_init": self.w_init, "w_star": self.w_star }
	
# 	def set_params( self, **params ):
# 		for key, value in params.items():
# 			setattr( self, key, value )
	
	def fit( self, X, y, max_iter = 50 ):
		start_time = time.time()
		n, d = X.shape
		
		batch_size = int( n / max_iter )
		batches = [ np.arange( batch_size ) + i * batch_size for i in range( max_iter ) ]
		
		self.l2=[]
		self.clock=[]
		self.w = self.w_init
		# np.random.normal( 0,1,(d,1) ) / np.sqrt( d )
		
		for i in range( max_iter ):
			self.l2.append( np.linalg.norm( self.w - self.w_star ) )
			grad = 2 * X[ batches[i] ] * ( np.matmul( X[ batches[i] ], self.w ) - y[ batches[i] ] ).reshape( -1, 1 )
			rgrad = RobustGrad(grad, self.alpha, self.delta)
			self.w = self.w - self.eta * rgrad
			self.clock.append(time.time()-start_time)
		
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )