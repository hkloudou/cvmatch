module github.com/hkloudou/cvmatch/bench

go 1.24.0

toolchain go1.24.7

require (
	github.com/hkloudou/cv2 v0.41200.0
	github.com/hkloudou/cvmatch v0.0.0
)

require (
	github.com/hkloudou/cv2/libs/darwin_arm64 v0.41200.0 // indirect
	github.com/hkloudou/cv2/libs/linux_386 v0.41200.0 // indirect
	github.com/hkloudou/cv2/libs/linux_amd64 v0.41200.0 // indirect
	github.com/hkloudou/cv2/libs/linux_arm64 v0.41200.0 // indirect
	github.com/hkloudou/cv2/libs/windows_386 v0.41200.0 // indirect
	github.com/hkloudou/cv2/libs/windows_amd64 v0.41200.0 // indirect
)

replace github.com/hkloudou/cvmatch => ../
