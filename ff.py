import pygmo as pg
import numpy as np

class fruit_fly:
    def __init__(self, tolerance=1e-5, max_iterations = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.x_0 = None
        self.f_0 = np.inf
        self.iterations = 0
        
    def step(self, lower_bound, upper_bound):
        radius = (upper_bound-lower_bound)/2
        progress = self.iterations/self.max_iterations
        step_scale = radius*np.exp(np.log(self.tolerance)*progress)
                
        return step_scale*np.random.uniform(-1,1)
    
    def update(self, pop):
        f_list = pop.get_f()
        i_best = np.argmin(f_list)
        x_best = pop.get_x()[i_best]
        f_best = f_list[i_best]
        if f_best<self.f_0:
            self.x_0 = x_best
            self.f_0 = f_best

    def evolve(self, pop):
        if self.iterations == 0:
            self.update(pop)
        bounds = pop.problem.get_bounds()
        for i, x in enumerate(pop.get_x()):
            d = np.random.randint(x.size)
            x_new = self.x_0.copy()
            x_new[d] = x_new[d]+self.step(bounds[0][d],bounds[1][d])
            pop.set_x(i, x_new.clip(bounds[0], bounds[1]))
        self.update(pop)
        self.iterations += 1
        
        return pop
    
    def get_best(self):
        return (self.x_sol, self.y_sol)