(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g k j a h)
(:init 
(harmony)
(planet g)
(planet k)
(planet j)
(planet a)
(planet h)
(province g)
(province k)
(province j)
(province a)
(province h)
)
(:goal
(and
(craves g k)
(craves k j)
(craves j a)
(craves a h)
)))