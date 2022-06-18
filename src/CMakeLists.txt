include(FetchContent)
FetchContent_Declare(
  guass_utils
  GIT_REPOSITORY https://github.com/andreacasalino/Gaussian-Utils.git
  GIT_TAG        58b3f207cf6df73a0b7cdc8bde933b06d20ab502
)
FetchContent_MakeAvailable(guass_utils)

set(PROJECT_SHORTNAME GaussianMixtureModels)

MakeLibrary(${PROJECT_SHORTNAME} header)

target_link_libraries(${PROJECT_SHORTNAME} PUBLIC
GaussianUtils
)

