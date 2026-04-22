import SwiftUI
import SceneKit

/// Renders an SMPL body mesh using SceneKit.
/// Updates geometry each frame from GVHMR-predicted vertices.
struct Mesh3DView: UIViewRepresentable {
    let vertices: [SIMD3<Float>]?
    let faces: [UInt32]   // flat [v0, v1, v2, v0, v1, v2, ...]
    /// Multi-person meshes: array of (vertices, trackID, translation) tuples.
    /// Translation is in camera coords (Y-down, Z-forward); converted to SceneKit internally.
    var multiPersonMeshes: [(vertices: [SIMD3<Float>], trackID: Int, translation: SIMD3<Float>?)]?

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.backgroundColor = UIColor(white: 0.1, alpha: 1)
        scnView.allowsCameraControl = true   // pinch/rotate/pan
        scnView.autoenablesDefaultLighting = false
        scnView.antialiasingMode = .multisampling4X

        let scene = SCNScene()
        scnView.scene = scene

        // Camera
        let cameraNode = SCNNode()
        cameraNode.name = "camera"
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.zNear = 0.01
        cameraNode.camera?.zFar = 20
        cameraNode.camera?.fieldOfView = 45
        cameraNode.position = SCNVector3(0, 0, 3)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)

        // Lighting: key + fill + ambient
        let keyLight = SCNNode()
        keyLight.light = SCNLight()
        keyLight.light?.type = .directional
        keyLight.light?.intensity = 800
        keyLight.light?.color = UIColor.white
        keyLight.position = SCNVector3(2, 3, 4)
        keyLight.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(keyLight)

        let fillLight = SCNNode()
        fillLight.light = SCNLight()
        fillLight.light?.type = .directional
        fillLight.light?.intensity = 400
        fillLight.light?.color = UIColor(white: 0.9, alpha: 1)
        fillLight.position = SCNVector3(-2, 1, 3)
        fillLight.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(fillLight)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 200
        ambient.light?.color = UIColor(white: 0.5, alpha: 1)
        scene.rootNode.addChildNode(ambient)

        // Mesh node placeholder
        let meshNode = SCNNode()
        meshNode.name = "smplMesh"
        scene.rootNode.addChildNode(meshNode)

        // Label
        let labelNode = context.coordinator.makeLabelNode()
        scene.rootNode.addChildNode(labelNode)

        return scnView
    }

    func updateUIView(_ scnView: SCNView, context: Context) {
        guard let scene = scnView.scene else { return }

        if let personMeshes = multiPersonMeshes, !personMeshes.isEmpty {
            // Multi-person: hide single mesh, show per-person nodes
            scene.rootNode.childNode(withName: "smplMesh", recursively: false)?.geometry = nil

            let existingNames = Set(scene.rootNode.childNodes
                .compactMap { $0.name }
                .filter { $0.hasPrefix("person_") })
            var neededNames = Set<String>()

            for (verts, trackID, translation) in personMeshes {
                let nodeName = "person_\(trackID)"
                neededNames.insert(nodeName)

                let node: SCNNode
                if let existing = scene.rootNode.childNode(withName: nodeName, recursively: false) {
                    node = existing
                } else {
                    node = SCNNode()
                    node.name = nodeName
                    scene.rootNode.addChildNode(node)
                }

                // Apply translation to separate persons in 3D space
                // Camera coords (Y-down, Z-forward) → SceneKit (Y-up, Z-toward-viewer)
                if let t = translation {
                    node.position = SCNVector3(t.x, -t.y, -t.z)
                } else {
                    node.position = SCNVector3Zero
                }

                let rgb = PersonColors.color(for: trackID)
                let personColor = UIColor(red: CGFloat(rgb.0), green: CGFloat(rgb.1), blue: CGFloat(rgb.2), alpha: 1)
                node.geometry = buildGeometry(vertices: verts, faces: faces, meshColor: personColor)
            }

            // Remove nodes for persons no longer present
            for name in existingNames where !neededNames.contains(name) {
                scene.rootNode.childNode(withName: name, recursively: false)?.removeFromParentNode()
            }
        } else {
            // Single person mode — remove leftover person nodes
            scene.rootNode.childNodes
                .filter { $0.name?.hasPrefix("person_") == true }
                .forEach { $0.removeFromParentNode() }

            guard let meshNode = scene.rootNode.childNode(withName: "smplMesh", recursively: false) else { return }
            guard let verts = vertices, !verts.isEmpty else {
                meshNode.geometry = nil
                return
            }
            meshNode.geometry = buildGeometry(vertices: verts, faces: faces)
        }
    }

    // MARK: - Geometry Builder

    private func buildGeometry(vertices: [SIMD3<Float>], faces: [UInt32], meshColor: UIColor? = nil) -> SCNGeometry {
        // Vertex positions
        let vertexData = Data(bytes: vertices, count: vertices.count * MemoryLayout<SIMD3<Float>>.stride)
        let vertexSource = SCNGeometrySource(
            data: vertexData,
            semantic: .vertex,
            vectorCount: vertices.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride
        )

        // Compute per-vertex normals
        let normals = computeNormals(vertices: vertices, faces: faces)
        let normalData = Data(bytes: normals, count: normals.count * MemoryLayout<SIMD3<Float>>.stride)
        let normalSource = SCNGeometrySource(
            data: normalData,
            semantic: .normal,
            vectorCount: normals.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride
        )

        // Triangle indices
        let indexData = Data(bytes: faces, count: faces.count * MemoryLayout<UInt32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: faces.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )

        let geometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])

        // Material: skin-like appearance (or per-person color)
        let material = SCNMaterial()
        material.diffuse.contents = meshColor ?? UIColor(red: 0.55, green: 0.45, blue: 0.4, alpha: 1)
        material.specular.contents = UIColor(white: 0.3, alpha: 1)
        material.shininess = 0.3
        material.lightingModel = .blinn
        material.isDoubleSided = true
        geometry.materials = [material]

        return geometry
    }

    /// Compute smooth vertex normals by averaging face normals.
    private func computeNormals(vertices: [SIMD3<Float>], faces: [UInt32]) -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: vertices.count)
        let numFaces = faces.count / 3

        for f in 0..<numFaces {
            let i0 = Int(faces[f * 3])
            let i1 = Int(faces[f * 3 + 1])
            let i2 = Int(faces[f * 3 + 2])

            let v0 = vertices[i0]
            let v1 = vertices[i1]
            let v2 = vertices[i2]

            let e1 = v1 - v0
            let e2 = v2 - v0
            let fn = cross(e1, e2)   // area-weighted face normal

            normals[i0] += fn
            normals[i1] += fn
            normals[i2] += fn
        }

        for i in 0..<normals.count {
            let len = length(normals[i])
            if len > 1e-8 {
                normals[i] /= len
            }
        }
        return normals
    }

    // MARK: - Coordinator

    func makeCoordinator() -> Coordinator { Coordinator() }

    class Coordinator {
        func makeLabelNode() -> SCNNode {
            let text = SCNText(string: "SMPL Mesh", extrusionDepth: 0)
            text.font = UIFont.systemFont(ofSize: 0.06, weight: .semibold)
            text.firstMaterial?.diffuse.contents = UIColor.white.withAlphaComponent(0.6)
            let node = SCNNode(geometry: text)
            node.position = SCNVector3(-0.15, 1.1, 0)
            return node
        }
    }
}
