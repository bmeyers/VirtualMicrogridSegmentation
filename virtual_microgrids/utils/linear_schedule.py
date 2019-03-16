class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t):
        """
        Updates epsilon

        Args:
            t: int
                frame number
        """
        if t <= self.nsteps:
            self.epsilon = self.eps_begin + (self.eps_end - self.eps_begin) * t / self.nsteps
        else:
            self.epsilon = self.eps_end