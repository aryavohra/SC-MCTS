(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g k j c f)
(:init 
(harmony)
(planet g)
(planet k)
(planet j)
(planet c)
(planet f)
(province g)
(province k)
(province j)
(province c)
(province f)
)
(:goal
(and
(craves g k)
(craves k j)
(craves j c)
(craves c f)
)))