@inline function find_next_intersection(x₀::T, y₀::T, x₁::T, y₁::T) where T

	f(t) = (x₀, y₀) .+ t .* (x₁ - x₀, y₁ - y₀)

	if x₁ >= x₀
		if (ceil(x₀) - x₀) != 0
			ts_x = (ceil(x₀) - x₀) / (x₁ - x₀)
		else
			ts_x = (ceil(x₀) - x₀ + 1) / (x₁ - x₀)
		end
	else
		if (floor(x₀) - x₀) != 0
			ts_x = (floor(x₀) - x₀) / (x₁ - x₀)
		else
			ts_x = (floor(x₀) - x₀ - 1) / (x₁ - x₀)
		end	
	end

	if y₁ >= y₀
		if (ceil(y₀) - y₀) != 0
			ts_y = (ceil(y₀) .- y₀) ./ (y₁ .- y₀)
		else			
			ts_y = (ceil(y₀) .- y₀ + 1) ./ (y₁ .- y₀)
		end
	else
		if (floor(y₀) - y₀) != 0
			ts_y = (floor(y₀) - y₀) / (y₁ - y₀)
		else
			ts_y = (floor(y₀) - y₀ - 1) / (y₁ - y₀)
		end	
	end

	# some weird edge cases
	if ts_y < ts_x && ts_y != T(Inf)
		return f(ts_y)
	else
		return f(ts_x)
	end
end


# based on this: https://cs.stackexchange.com/questions/132887/determining-the-intersections-of-a-line-segment-and-grid
function find_intersections(x₀::T, y₀::T, x₁::T, y₁::T) where T
	if x₀ ≈ x₁ && y₀ ≈ y₁
		return Array{T, 2}(undef, 2, 0)
	end

	f(t) = (x₀, y₀) .+ t .* (x₁ - x₀, y₁ - y₀)

	# avoid division by zero
	if x₁ != x₀
		if x₁ >= x₀
			ts_x = ((ceil(x₀):floor(x₁)) .- x₀) ./ (x₁ .- x₀)
		else
			ts_x = ((floor(x₀):-1:ceil(x₁)) .- x₀) ./ (x₁ .- x₀)
		end
	else
		ts_x = T[]
	end

	if y₁ != y₀
		if y₁ >= y₀
			ts_y = ((ceil(y₀):floor(y₁)) .- y₀) ./ (y₁ .- y₀)
		else
			ts_y = ((floor(y₀):-1:ceil(y₁)) .- y₀) ./ (y₁ .- y₀)
		end
	else
		ts_y = T[]
	end

	i, j = 1, 1
	points = Array{T, 2}(undef, 2, length(ts_x) + length(ts_y))

	# do a sorted merge of the twlists
	while i ≤ length(ts_x) && j ≤ length(ts_y)
		if ts_x[i] < ts_y[j]
			points[:, i + j - 1] .= f(ts_x[i])
			i += 1
		else
			points[:, i + j - 1] .= f(ts_y[j])
			j += 1
		end
	end

	# if one of the lists has leftovers, add them
	while j ≤ length(ts_y)
		points[:, i + j - 1] .= f(ts_y[j])
		j += 1
	end

	while i ≤ length(ts_x)
		points[:, i + j - 1] .= f(ts_x[i])
		i += 1
	end

	# some points occur multiple times, filter them out
	if points[:, begin] == points[:, begin + 1]
		startoffset = 1
	else
		startoffset = 0
	end

	if points[:, end] == points[:, end - 1]
		endoffset = 1
	else
		endoffset = 0
	end

	return points[:, begin + startoffset : end - endoffset]
end


"""
	find_cell(a, b)

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
@inline function find_cell(a, b)
	T = eltype(a)
	x = floor(Int, (b[1] + a[1]) / 2)
	y = floor(Int, (b[2] + a[2]) / 2)
	return x, y
end


@inline function next_ray_on_circle(img, angle, y_dist, y_dist_end, mid, radius, sinα, cosα)
	x_dist = sqrt(radius^2 - y_dist^2)
    x_dist_end = y_dist == y_dist_end ? -x_dist : -sqrt(radius^2 - y_dist_end^2)
	y_dist_end = y_dist_end
	
	x_dist_rot = mid + cosα * x_dist - sinα * y_dist 
	y_dist_rot = mid + sinα * x_dist + cosα * y_dist

	x_dist_end_rot = mid + cosα * x_dist_end - sinα * y_dist_end
	y_dist_end_rot = mid + sinα * x_dist_end + cosα * y_dist_end
	return x_dist_rot, y_dist_rot, x_dist_end_rot, y_dist_end_rot
end


function rays_on_circle(img, angle)
	radius = size(img, 1) ÷ 2 - 1
	mid = size(img, 1) ÷ 2 + 1.5f0

	y_dists = similar(img, (size(img, 1) - 1))
	y_dists .= -radius:radius
	y_dists_end = copy(y_dists)

	x_dists = similar(img, (size(img, 1) - 1))
	x_dists .= .- sqrt.(radius .^2 .- y_dists .^2)
	x_dists_end = .- x_dists

	# reduce memory
	# rotate the correct coordinates around center
	buffer = copy(x_dists)
	buffer .= cos.(angle) .* x_dists .- sin.(angle) .* y_dists
	y_dists .= mid .+ sin.(angle) .* x_dists .+ cos.(angle) .* y_dists
	x_dists .= mid .+ buffer

	buffer .= cos.(angle) .* x_dists_end .- sin.(angle) .* y_dists_end
	y_dists_end .=  mid .+ sin.(angle) .* x_dists_end .+ cos.(angle) .* y_dists_end
	x_dists_end .= mid .+ buffer

	return x_dists, y_dists, x_dists_end, y_dists_end
end

