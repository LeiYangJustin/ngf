#pragma once

#include <tuple>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "microlog.h"

#define GLSL_ALIGN alignas(sizeof(glm::vec4))

struct RayFrame {
	GLSL_ALIGN glm::vec3 origin;
	GLSL_ALIGN glm::vec3 lower_left;
	GLSL_ALIGN glm::vec3 horizontal;
	GLSL_ALIGN glm::vec3 vertical;
};

struct Transform {
	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);

	void from(const glm::vec3 &, const glm::vec3 &, const glm::vec3 &);

	glm::mat4 matrix() const;

	glm::vec3 right() const;
	glm::vec3 up() const;
	glm::vec3 forward() const;

	std::tuple <glm::vec3, glm::vec3, glm::vec3> axes() const;
};

struct Camera {
	float aspect = 1.0f;
	float fov = 45.0f;
	float near = 0.1f;
	float far = 1000.0f;

	void from(float, float = 45.0f, float = 0.1f, float = 1000.0f);
	glm::mat4 perspective_matrix() const;
	RayFrame rayframe(const Transform &) const;

	static glm::mat4 view_matrix(const Transform &);
};
