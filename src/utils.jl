"""
	find_cell(x, y, xnew, ynew)

Find the correct intersected cell.
`a` and `b` are a Vector of length 2

Cells are identified by the coordinates in the lower left
corner of the cell.


|---------------------------|
| 		 |         |        |
|(3,1)   |(3,2)    |(3,3)   |
|--------|---------|--------|
|        |         |        |
|(2,1)   |(2,2)    |(2,3)   |
|--------|---------|--------|
|        |         |        |
|(1,1)   |(1,2)    |(1,3)   |
|--------|---------|--------|
"""
@inline function find_cell(x, y, xnew, ynew)
	x = floor(Int, (x + xnew) / 2)
	y = floor(Int, (y + ynew) / 2)
	return x, y
end


@inline function next_cell_intersection(x₀, y₀, x₁, y₁)
	# inspired by 
	floor_or_ceilx = x₁ ≥ x₀ ? ceil : floor
	floor_or_ceily = y₁ ≥ y₀ ? ceil : floor
	# https://cs.stackexchange.com/questions/132887/determining-the-intersections-of-a-line-segment-and-grid
	txx = (floor_or_ceilx(x₀)-x₀)
    xdiff = x₁ - x₀
    ydiff = y₁ - y₀
    txx = ifelse(txx != 0, txx, txx + sign(xdiff))
	tyy = (floor_or_ceily(y₀)-y₀)
    tyy = ifelse(tyy != 0, tyy, tyy + sign(ydiff))

	tx = txx / (xdiff)
	ty = tyy / (ydiff)
	# decide which t is smaller, and hence the next step
	t = ifelse(tx > ty, ty, tx)

	# calculate new coordinates
	x = x₀ + t * (xdiff)
	y = y₀ + t * (ydiff)
    return x, y, sign(xdiff), sign(ydiff)
end


@inline function next_ray_on_circle(y_dist::T, y_dist_end, mid, radius, sinα, cosα) where T 
	x_dist = sqrt(radius^2 - y_dist^2) + T(0.5)
    x_dist_end = -sqrt(radius^2 - y_dist_end^2)
	y_dist_end = y_dist_end
	
	x_dist_rot = mid + cosα * x_dist - sinα * y_dist 
	y_dist_rot = mid + sinα * x_dist + cosα * y_dist

	x_dist_end_rot = mid + cosα * x_dist_end - sinα * y_dist_end
	y_dist_end_rot = mid + sinα * x_dist_end + cosα * y_dist_end
	return x_dist_rot, y_dist_rot, x_dist_end_rot, y_dist_end_rot
end

